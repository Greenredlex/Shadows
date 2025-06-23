import gradio as gr
import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium import plugins
import tempfile
import json
from datetime import datetime, time, timedelta
import pytz
import pvlib
from PIL import Image
import io
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
import traceback

# Import the road segment splitter
try:
    from road_segment_splitter import load_split_segments_for_shadow_analysis
    SPLIT_SEGMENTS_AVAILABLE = True
except ImportError:
    print("⚠️ Road segment splitter not available - using original road loading method")
    SPLIT_SEGMENTS_AVAILABLE = False

# Import original calculation functions
def check_files_and_modules():
    """Check if all required files and modules are available"""
    status = {"files": [], "modules": [], "ready": True}
    
    # Check files
    required_files = ['2023_R_25GN1.TIF', 'bgt_wegdeel.gml']
    for file in required_files:
        if os.path.exists(file):
            status["files"].append(f"✅ {file}")
        else:
            status["files"].append(f"❌ {file} NIET GEVONDEN")
            status["ready"] = False
    
    # Check optional split segments file
    if os.path.exists('split_bike_foot_paths.gpkg'):
        status["files"].append(f"✅ split_bike_foot_paths.gpkg (50m segmenten)")
    else:
        status["files"].append(f"ℹ️ split_bike_foot_paths.gpkg (wordt automatisch aangemaakt)")
    
    # Check road segment splitter
    if SPLIT_SEGMENTS_AVAILABLE:
        status["files"].append(f"✅ road_segment_splitter.py (betere resolutie)")
    else:
        status["files"].append(f"ℹ️ road_segment_splitter.py (optioneel)")
    
    # Check modules
    required_modules = ['rasterio', 'geopandas', 'pvlib', 'folium', 'contextily', 'numba']
    for module in required_modules:
        try:
            __import__(module)
            status["modules"].append(f"✅ {module}")
        except ImportError:
            status["modules"].append(f"❌ {module} NIET GEVONDEN")
            status["ready"] = False
    
    return status

def get_default_bbox():
    """Get a reasonable default bounding box"""
    try:
        with rasterio.open('2023_R_25GN1.TIF') as src:
            bounds = src.bounds
            if str(src.crs) != 'EPSG:4326':
                import pyproj
                transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
                max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
                # Take a small central area
                center_lon, center_lat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
                delta = 0.002  # Small area
                return f"{center_lon-delta},{center_lat-delta},{center_lon+delta},{center_lat+delta}"
    except:
        pass
    return "4.885,52.365,4.890,52.370"

def complete_shadow_analysis(time_step_minutes, start_hour, end_hour, object_height_threshold, 
                            analysis_date, bbox_coords, progress=gr.Progress()):
    """
    Complete shadow analysis with all visualizations. 
    Yields log updates in real-time and returns all plots at the very end.
    """
    # Variabelen om de plots op te slaan tot het einde
    dsm_img_out = None
    sunpath_img_out = None
    pixel_heatmap_img_out = None
    road_heatmap_img_out = None
    
    global SPLIT_SEGMENTS_AVAILABLE
    
    log_accumulator = []
    
    def yield_log(message):
        """Helper to yield ONLY a log update, telling Gradio to not touch the image components."""
        if message:
            log_accumulator.append(message)
        # gr.update() is the command to "leave this component as is"
        return (gr.update(), gr.update(), gr.update(), gr.update(), "\n".join(log_accumulator))

    try:
        from numba import jit
        import rasterio.features
        from shapely.geometry import mapping
        
        @jit(nopython=True)
        def calculate_shadows(dsm_data, nodata_value, transform, sun_azimuth_rad, sun_elevation_rad, object_mask):
            rows, cols = dsm_data.shape
            shadow_mask = np.full(dsm_data.shape, 1.0, dtype=np.float32)
            pixel_res = abs(transform[0])
            dx = np.sin(sun_azimuth_rad)
            dy = -np.cos(sun_azimuth_rad)
            tan_elevation = np.tan(sun_elevation_rad)
            max_shadow_dist_pixels = int(200 / pixel_res) if pixel_res > 0 else 100

            for r in range(rows):
                for c in range(cols):
                    h_pixel = dsm_data[r, c]
                    if np.isnan(h_pixel):
                        shadow_mask[r, c] = np.nan
                        continue
                    if object_mask[r, c] == 1:
                        shadow_mask[r, c] = 0.0
                        continue
                    if tan_elevation <= 0:
                        shadow_mask[r, c] = 0.0
                        continue
                    for i in range(1, max_shadow_dist_pixels + 1):
                        check_c = c + int(round(i * dx))
                        check_r = r + int(round(i * dy))
                        if not (0 <= check_r < rows and 0 <= check_c < cols):
                            break
                        dist_pixels = np.sqrt((check_c - c)**2 + (check_r - r)**2)
                        dist_meters = dist_pixels * pixel_res
                        min_blocker_height = h_pixel + dist_meters * tan_elevation
                        h_check = dsm_data[check_r, check_c]
                        if np.isnan(h_check):
                            continue
                        if h_check > min_blocker_height:
                            shadow_mask[r, c] = 0.0
                            break
            return shadow_mask
        
        progress(0.05, "🚀 Start complete analyse...")
        yield yield_log("🚀 Start complete analyse...")
        
        if isinstance(analysis_date, str):
            analysis_date = datetime.strptime(analysis_date, '%Y-%m-%d').date()
        
        bbox_parsed = None
        if bbox_coords and bbox_coords.strip():
            try:
                coords = [float(x.strip()) for x in bbox_coords.split(',')]
                if len(coords) == 4:
                    bbox_parsed = coords
                    yield yield_log(f"📍 Bounding box: {bbox_parsed}")
            except:
                yield yield_log("⚠️ Bbox parse fout, gebruik default")
        
        progress(0.1, "📊 DSM data laden en normaliseren...")
        
        with rasterio.open('2023_R_25GN1.TIF') as src:
            dsm_crs = src.crs
            nodata_value = src.nodata if src.nodata is not None else -9999.0
            
            if bbox_parsed:
                import pyproj
                transformer = pyproj.Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
                min_x, min_y = transformer.transform(bbox_parsed[0], bbox_parsed[1])
                max_x, max_y = transformer.transform(bbox_parsed[2], bbox_parsed[3])
                bbox_native = [min_x, min_y, max_x, max_y]
                window = rasterio.windows.from_bounds(*bbox_native, src.transform)
            else:
                window = rasterio.windows.Window(0, 0, min(1500, src.width), min(1500, src.height))
            
            dsm_subset = src.read(1, window=window)
            transform = src.window_transform(window)
            bounds = rasterio.windows.bounds(window, src.transform)
            
            dsm_subset_float = dsm_subset.astype(np.float32)
            nodata_mask = (dsm_subset == nodata_value)
            dsm_subset_nan = dsm_subset_float.copy()
            dsm_subset_nan[nodata_mask] = np.nan
            
            dsm_for_shadow_calc = dsm_subset_float.copy()
            min_valid_height = np.nanmin(dsm_subset_nan)
            water_height = min_valid_height - 1.0
            dsm_for_shadow_calc[nodata_mask] = water_height
            
            dsm_data = dsm_for_shadow_calc - min_valid_height
            yield yield_log(f"📊 DSM genormaliseerd: min_hoogte={min_valid_height:.1f}m, water_hoogte={water_height:.1f}m")
            
            object_mask = (dsm_data > object_height_threshold).astype(np.uint8)
            yield yield_log(f"🏗️ Object masker: {np.sum(object_mask)} pixels > {object_height_threshold}m")
        
        yield yield_log(f"📐 DSM geladen: {dsm_data.shape}")
        
        progress(0.15, "🛣️ Wegen data laden en filteren...")
        
        if SPLIT_SEGMENTS_AVAILABLE:
            try:
                bbox_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
                bike_foot_paths_clipped = load_split_segments_for_shadow_analysis('split_bike_foot_paths.gpkg', bbox_bounds)
                if bike_foot_paths_clipped.crs != dsm_crs:
                    bike_foot_paths_clipped = bike_foot_paths_clipped.to_crs(dsm_crs)
                yield yield_log(f"🚴 Fietspaden/voetpaden (50m segmenten): {len(bike_foot_paths_clipped)} segmenten")
                yield yield_log(f"   ✅ Gebruikmakend van vooraf gesplitste segmenten voor betere resolutie")
            except Exception as e:
                yield yield_log(f"⚠️ Fout bij laden gesplitste segmenten: {e}")
                yield yield_log("   Terugvallen op originele methode...")
                SPLIT_SEGMENTS_AVAILABLE = False
        
        if not SPLIT_SEGMENTS_AVAILABLE:
            roads_gdf = gpd.read_file('bgt_wegdeel.gml')
            if 'function' in roads_gdf.columns:
                bike_foot_paths = roads_gdf[roads_gdf['function'].isin(['fietspad', 'voetpad'])]
                if len(bike_foot_paths) == 0: bike_foot_paths = roads_gdf.head(500)
            else: bike_foot_paths = roads_gdf.head(500)
            if bike_foot_paths.crs != dsm_crs: bike_foot_paths = bike_foot_paths.to_crs(dsm_crs)
            from shapely.geometry import box
            clip_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
            bike_foot_paths_clipped = bike_foot_paths.clip(clip_box)
            yield yield_log(f"🚴 Fietspaden/voetpaden (origineel): {len(bike_foot_paths_clipped)} segmenten")
        
        progress(0.2, "⏰ Tijdreeks maken...")
        local_tz = pytz.timezone('Europe/Amsterdam')
        start_dt = local_tz.localize(datetime.combine(analysis_date, time(start_hour, 0)))
        end_dt = local_tz.localize(datetime.combine(analysis_date, time(end_hour, 0)))
        time_range = pd.date_range(start=start_dt, end=end_dt, freq=f'{time_step_minutes}min')
        yield yield_log(f"🕐 Tijdstappen: {len(time_range)}")
        
        shadow_accumulator = np.zeros_like(dsm_data, dtype=np.float32)
        valid_times = 0
        latitude, longitude = 52.37, 4.89
        sun_positions = []
        
        for i, current_dt in enumerate(time_range):
            step_progress = 0.2 + 0.4 * (i / len(time_range))
            progress(step_progress, f"☀️ Schaduw berekenen: Tijdstap {i+1}/{len(time_range)} ({current_dt.strftime('%H:%M')})")
            
            current_dt_utc = current_dt.astimezone(pytz.utc)
            loc = pvlib.location.Location(latitude, longitude, tz='UTC')
            solar_pos = loc.get_solarposition(pd.to_datetime([current_dt_utc]))
            sun_azimuth, sun_elevation = solar_pos['azimuth'].iloc[0], solar_pos['apparent_elevation'].iloc[0]
            sun_positions.append({'time': current_dt.strftime('%H:%M'), 'azimuth': sun_azimuth, 'elevation': sun_elevation})
            
            if sun_elevation > 0:
                shadow_result = calculate_shadows(dsm_data, np.nan, transform, np.radians(sun_azimuth), np.radians(sun_elevation), object_mask)
                shadow_accumulator += shadow_result
                valid_times += 1
                yield yield_log(f"✅ {current_dt.strftime('%H:%M')}: Az={sun_azimuth:.1f}°, El={sun_elevation:.1f}° - Schaduw berekend")
            else:
                yield yield_log(f"⏳ {current_dt.strftime('%H:%M')}: Az={sun_azimuth:.1f}°, El={sun_elevation:.1f}° - Zon onder horizon")
        
        shadow_percentage = (1 - (shadow_accumulator / valid_times)) * 100 if valid_times > 0 else np.zeros_like(dsm_data)
        
        progress(0.65, "🗺️ Visualisaties maken...")
        
        # === 1. DSM Preview ===
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.set_xlim(bounds[0], bounds[2]); ax1.set_ylim(bounds[1], bounds[3]); ax1.set_aspect('equal')
        try: ctx.add_basemap(ax1, crs=dsm_crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.6)
        except Exception as e: yield yield_log(f"⚠️ Basemap kon niet worden toegevoegd: {str(e)}")
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
        im1 = ax1.imshow(dsm_subset_nan, cmap='terrain', extent=extent, alpha=0.8)
        object_overlay = np.ma.masked_where(object_mask == 0, object_mask)
        ax1.imshow(object_overlay, extent=extent, cmap='Reds', alpha=0.4, vmin=0, vmax=1)
        plt.colorbar(im1, ax=ax1, label='Hoogte (m NAP)')
        ax1.set_title(f'Digital Surface Model (DSM)\nObjecten > {object_height_threshold}m (rood overlay)', y=1.03)
        img1_buffer = io.BytesIO(); plt.savefig(img1_buffer, format='png', dpi=150, bbox_inches='tight'); img1_buffer.seek(0)
        dsm_img_out = Image.open(img1_buffer)
        plt.close(fig1)
        yield yield_log("🖼️ DSM plot gegenereerd.")

        # === 2. Sun Path ===
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.plot([p['azimuth'] for p in sun_positions], [p['elevation'] for p in sun_positions], 'o-', lw=3, ms=8, color='orange', label='Zonnepad')
        ax2.set_xlabel('Azimut (°)', fontsize=12); ax2.set_ylabel('Elevatie (°)', fontsize=12)
        ax2.set_title(f'Zonnepad - {analysis_date}', fontsize=14, fontweight='bold', y=1.03); ax2.grid(True, alpha=0.3); ax2.legend()
        img2_buffer = io.BytesIO(); plt.savefig(img2_buffer, format='png', dpi=150, bbox_inches='tight'); img2_buffer.seek(0)
        sunpath_img_out = Image.open(img2_buffer)
        plt.close(fig2)
        yield yield_log("☀️ Zonnepad plot gegenereerd.")
        
        progress(0.75, "📈 Pixel-level heatmap maken...")
        
        # === 3. Pixel-level Shadow Heatmap ===
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        cmap = LinearSegmentedColormap.from_list('shadow_cmap', ['red', 'orange', 'yellow', 'lightblue', 'blue', 'darkblue'], N=100)
        ax3.set_xlim(bounds[0], bounds[2]); ax3.set_ylim(bounds[1], bounds[3]); ax3.set_aspect('equal')
        try: ctx.add_basemap(ax3, crs=dsm_crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
        except Exception as e: yield yield_log(f"⚠️ Basemap pixel heatmap kon niet worden toegevoegd: {str(e)}")
        im3 = ax3.imshow(shadow_percentage, extent=extent, cmap=cmap, vmin=0, vmax=100, alpha=0.8)
        if len(bike_foot_paths_clipped) > 0: bike_foot_paths_clipped.plot(ax=ax3, color='white', linewidth=1)
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8); cbar3.set_label('Gemiddeld % tijd in schaduw', fontsize=12)
        ax3.set_title(f'Pixel-level Schaduw Heatmap ({start_hour:02d}:00-{end_hour:02d}:00)', fontsize=14, fontweight='bold', y=1.03)
        img3_buffer = io.BytesIO(); plt.savefig(img3_buffer, format='png', dpi=180, bbox_inches='tight'); img3_buffer.seek(0)
        pixel_heatmap_img_out = Image.open(img3_buffer)
        plt.close(fig3)
        yield yield_log("🌡️ Pixel heatmap gegenereerd.")
        
        progress(0.85, "🚴 Per-fietspad analyse maken...")
        
        # === 4. Per-Road Shadow Analysis ===
        if len(bike_foot_paths_clipped) > 0:
            road_shadows = []
            for _, road in bike_foot_paths_clipped.iterrows():
                try:
                    road_mask = rasterio.features.rasterize([mapping(road.geometry)], out_shape=dsm_data.shape, transform=transform, fill=0, default_value=1, dtype=np.uint8)
                    road_pixels = shadow_percentage[road_mask == 1]
                    road_shadows.append(np.nanmean(road_pixels) if len(road_pixels) > 0 else 0)
                except Exception: road_shadows.append(0)
            
            bike_foot_paths_clipped['shadow_pct'] = road_shadows
            fig4, ax4 = plt.subplots(figsize=(12, 10))
            road_cmap = LinearSegmentedColormap.from_list('road_shadow_cmap', ['red', 'orange', 'yellow', 'lightblue', 'blue', 'darkblue'], N=100)
            bike_foot_paths_clipped.plot(ax=ax4, column='shadow_pct', cmap=road_cmap, linewidth=3, vmin=0, vmax=100, legend=False)
            sm = plt.cm.ScalarMappable(cmap=road_cmap, norm=plt.Normalize(vmin=0, vmax=100)); sm.set_array([])
            cbar4 = plt.colorbar(sm, ax=ax4, shrink=0.8); cbar4.set_label('Gemiddeld % tijd in schaduw', fontsize=12)
            ax4.set_xlim(bounds[0], bounds[2]); ax4.set_ylim(bounds[1], bounds[3]); ax4.set_aspect('equal')
            try: ctx.add_basemap(ax4, crs=dsm_crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
            except Exception as e: yield yield_log(f"⚠️ Basemap per-road heatmap kon niet worden toegevoegd: {str(e)}")
            ax4.set_title(f'Gemiddelde Schaduw per Fietspad/Voetpad', fontsize=14, fontweight='bold', y=1.03)
            img4_buffer = io.BytesIO(); plt.savefig(img4_buffer, format='png', dpi=180, bbox_inches='tight'); img4_buffer.seek(0)
            road_heatmap_img_out = Image.open(img4_buffer)
            plt.close(fig4)
            yield yield_log("🚴 Per-fietspad plot gegenereerd.")
            yield yield_log(f"   Gemiddelde schaduw: {np.mean(road_shadows):.1f}%")
            yield yield_log(f"   Aantal segmenten: {len(road_shadows)}")
        else:
            yield yield_log("⚠️ Geen fietspaden/voetpaden gevonden voor per-segment analyse")
        
        progress(1.0, "✅ Complete analyse voltooid!")
        yield yield_log(f"📊 Totaal statistieken:")
        yield yield_log(f"   Pixel-level gemiddeld: {np.nanmean(shadow_percentage):.1f}%")
        yield yield_log(f"   Tijdstappen verwerkt: {valid_times}/{len(time_range)}")
        yield yield_log(f"🎉 Complete analyse voltooid! Alle resultaten worden nu getoond.")
        
        # --- DE FINALE YIELD ---
        # Geeft alle resultaten in één keer terug aan de UI.
        yield (dsm_img_out, sunpath_img_out, pixel_heatmap_img_out, road_heatmap_img_out, "\n".join(log_accumulator))
        
    except Exception as e:
        error_msg = f"❌ Complete analyse fout: {str(e)}\n{traceback.format_exc()}"
        log_accumulator.append(error_msg)
        # Stuur lege plots en de foutmelding in de logs
        yield (None, None, None, None, "\n".join(log_accumulator))


def create_complete_app():
    """Create the complete shadow analysis application"""
    
    # Check system status
    status = check_files_and_modules()
    
    with gr.Blocks(title="🌞 Complete Shadow Analysis", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🌞 Complete Shadow Analysis Tool
        **Interactieve schaduwanalyse met een vernieuwde, simpele kaartselectie.**
        """)
        
        # System status
        with gr.Row():
            with gr.Column():
                status_md = "## 🔍 Systeem Status\n\n"
                status_md += "### 📁 Bestanden:\n" + "\n".join([f"- {f}" for f in status["files"]])
                status_md += "\n\n### 📦 Modules:\n" + "\n".join([f"- {m}" for m in status["modules"]])
                if not status["ready"]:
                    status_md += "\n\n❌ **Systeem niet gereed voor analyse**"
                else:
                    status_md += "\n\n✅ **Systeem gereed voor analyse**"
                gr.Markdown(status_md)
        
        with gr.Row():
            # Left column: Parameters and map
            with gr.Column(scale=2):
                gr.Markdown("## ⚙️ Stap 1: Analyse Parameters Instellen")
                
                with gr.Row():
                    date_input = gr.Textbox(
                        label="📅 Datum (YYYY-MM-DD)", 
                        value=datetime.now().strftime('%Y-%m-%d'),
                        info="Datum voor schaduwanalyse"
                    )
                
                with gr.Row():
                    timestep_slider = gr.Slider(
                        label="⏱️ Tijdstap (minuten)", 
                        minimum=15, maximum=120, value=30,
                        info="Interval tussen berekeningen"
                    )
                
                with gr.Row():
                    start_hour_slider = gr.Slider(
                        label="🌅 Start uur", 
                        minimum=6, maximum=18, value=12,
                        info="Begin van analyse periode"
                    )
                    end_hour_slider = gr.Slider(
                        label="🌇 Eind uur", 
                        minimum=6, maximum=20, value=18,
                        info="Einde van analyse periode"
                    )
                
                threshold_slider = gr.Slider(
                    label="🏗️ Object drempel (meter)", 
                    minimum=1, maximum=20, value=10,
                    info="Minimale hoogte voor schaduw objecten (gebouwen, bomen)"
                )
                
                gr.Markdown("## 📍 Stap 2: Gebied Selecteren")
                
                # Create the map HTML with the new, simplified selection map
                map_content = create_selection_map()
                
                gr.HTML(
                    value=f"""
                    <div style="border: 2px solid #ddd; border-radius: 8px; overflow: hidden;">
                        <div style="background: #f8f9fa; padding: 10px; border-bottom: 1px solid #ddd;">
                            <h4 style="margin: 0; color: black;">🗺️ Kaart: Teken een Rechthoek</h4>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: black;">
                                <span style="color:red; font-weight: bold;">━━</span> DSM Data (Hoogte)
                                <span style="color:blue; font-weight: bold; margin-left: 10px;">━━</span> Wegen Data
                            </p>
                        </div>
                        <div style="height: 450px;">
                            {map_content}
                        </div>
                    </div>
                    """
                )

                bbox_input = gr.Textbox(
                    label="🎯 Geselecteerde Bounding Box (WGS84)",
                    placeholder="min_lon,min_lat,max_lon,max_lat",
                    value=get_default_bbox(),
                    info="Dit veld wordt automatisch bijgewerkt na het tekenen op de kaart.",
                    elem_id="bbox-input"  # Crucial ID for robust JS targeting
                )
                
                gr.Markdown("## 🚀 Stap 3: Analyse Uitvoeren")
                run_analysis_btn = gr.Button("🚀 Start Complete Analyse", variant="primary", size="lg")
            
            # Right column: Instructions
            with gr.Column(scale=1):
                gr.Markdown("""
                ## 💡 Instructies
                
                ### 1. **Parameters Instellen**
                Configureer de datum, tijdstap, periode en de objectdrempel voor de analyse.
                - **Object Drempel**: De minimale hoogte die een object moet hebben om een schaduw te werpen. Bijvoorbeeld 10 meter om de meeste bomen en gebouwen mee te nemen.
                
                ### 2. **Gebied Selecteren**
                Gebruik de kaart om het analysegebied te bepalen.
                - **Teken een rechthoek** met de (□) tool op de kaart. Kies bij voorkeur een gebied waar de rode en blauwe kaders overlappen.
                - Het **Bounding Box** veld wordt **automatisch ingevuld** zodra je klaar bent met tekenen. Je kunt de coördinaten ook handmatig invoeren of aanpassen.

                ### 3. **Analyse Starten**
                Klik op de **Start Complete Analyse** knop om de berekening uit te voeren. De resultaten verschijnen hieronder.

                ---
                
                ### 📊 Resultaten Dashboard
                De analyse genereert vier visualisaties:
                1.  **DSM + Objecten**: De hoogtekaart met de gedetecteerde schaduw-werpende objecten in het rood.
                2.  **Zonnepad**: Het pad van de zon gedurende de geselecteerde periode.
                3.  **Pixel Heatmap**: Een gedetailleerde kaart die per pixel toont hoeveel procent van de tijd deze in de schaduw ligt.
                4.  **Fietspad Analyse**: Een kaart die de gemiddelde hoeveelheid schaduw per fietspad- of voetpadsegment toont.
                """)
        
        # Results section
        gr.Markdown("## 📊 Analyse Resultaten")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Digital Surface Model (DSM)")
                dsm_img = gr.Image(type="pil", height=400, show_label=False)
                gr.Markdown("### 🌡️ Pixel-level Schaduw Heatmap") 
                pixel_heatmap_img = gr.Image(type="pil", height=400, show_label=False)
            with gr.Column():
                gr.Markdown("### ☀️ Zonnepad")
                sunpath_img = gr.Image(type="pil", height=400, show_label=False)
                gr.Markdown("### 🚴 Gemiddelde Schaduw per Fietspad/Voetpad")
                road_heatmap_img = gr.Image(type="pil", height=400, show_label=False)
        
        # Logs
        logs_output = gr.Textbox(
            label="📝 Analyse Logs", 
            lines=15, 
            interactive=False,
            info="Gedetailleerde voortgang en resultaten"
        )
        
        # JavaScript to link map selection to the textbox using postMessage
        gr.HTML("""
        <style>
            /* Simple animation to flash the background of an element */
            @keyframes flash-success {
                from { background-color: #c8e6c9; }
                to { background-color: transparent; }
            }
            .flash-on-update {
                animation: flash-success 1.5s ease-out;
            }
        </style>
        <script>
        // This function will be called when the map sends coordinates
        function updateBbox(coords) {
            console.log("Attempting to update bbox with:", coords);
            // Find the Gradio container for our textbox using its unique ID
            const bbox_container = document.querySelector("#bbox-input");
            
            if (!bbox_container) {
                console.error('❌ CRITICAL: Could not find the bbox container with elem_id: #bbox-input');
                alert('Fout: Kon het bounding box container-element niet vinden.');
                return;
            }

            // Find the actual <input> or <textarea> element inside the container
            const bbox_input_element = bbox_container.querySelector("textarea, input");
            
            if (bbox_input_element) {
                // Set the value
                bbox_input_element.value = coords;
                
                // Dispatch the 'input' event to notify Gradio of the change
                const inputEvent = new Event('input', { bubbles: true });
                bbox_input_element.dispatchEvent(inputEvent);
                
                // Add the CSS class to the container to trigger the flash animation
                bbox_container.classList.add('flash-on-update');
                // Remove the class after the animation is done to allow it to run again
                setTimeout(() => {
                    bbox_container.classList.remove('flash-on-update');
                }, 1500);

                console.log('✅ Bounding box updated successfully.');
            } else {
                console.error('❌ CRITICAL: Found the container, but not the input/textarea inside #bbox-input');
                alert('Fout: Kon het input-veld binnen de container niet vinden.');
            }
        }

        // Listen for 'message' events from the map iframe
        window.addEventListener('message', function(event) {
            if (event.data && event.data.type === 'bbox_update' && event.data.coordinates) {
                console.log('📡 Message received from map iframe:', event.data.coordinates);
                updateBbox(event.data.coordinates);
            }
        });
        </script>
        """)

        # Event handlers
        def run_complete_analysis_wrapper(date_str, timestep, start_h, end_h, threshold, bbox_str):
            """
            This wrapper consumes the generator from the main analysis function
            and returns only the final set of results.
            """
            # Start met lege outputs, vooral voor de logs
            outputs = gr.update(), gr.update(), gr.update(), gr.update(), ""
            
            # Loop door de generator heen. Elke 'yield' is een update.
            for outputs in complete_shadow_analysis(timestep, start_h, end_h, threshold, date_str, bbox_str):
                # We hoeven hier niets te doen, de loop consumeert de generator.
                # De 'outputs' variabele bevat na elke stap de laatste set resultaten.
                pass
                
            # Geef alleen de allerlaatste set resultaten terug die de generator heeft geproduceerd.
            return outputs
        
        run_analysis_btn.click(
            run_complete_analysis_wrapper,
            inputs=[date_input, timestep_slider, start_hour_slider, end_hour_slider, threshold_slider, bbox_input],
            outputs=[dsm_img, sunpath_img, pixel_heatmap_img, road_heatmap_img, logs_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **🌞 Complete Shadow Analysis Tool v2.1** - Met robuuste en eenvoudige kaartselectie.
        """)
    
    return app

# New, simplified map function for area selection
def create_selection_map():
    """
    Return HTML for a simple Leaflet map (via Folium) that lets the user draw one
    rectangle. The selected rectangle coordinates are posted to the parent window
    so the Gradio bounding-box field is automatically updated.
    - Red: DSM coverage
    - Blue: Roads/bike paths coverage
    - Orange: Your selection
    """
    try:
        # Determine coverage area of height (DSM) and road files
        with rasterio.open('2023_R_25GN1.TIF') as src:
            dsm_bounds = src.bounds
            dsm_crs = src.crs
        gml_gdf = gpd.read_file('bgt_wegdeel.gml')
        gml_bounds = gml_gdf.total_bounds

        # Convert to WGS84 (lat, lon) so Leaflet can draw the polygons correctly
        import pyproj
        if str(dsm_crs) != 'EPSG:4326':
            tf = pyproj.Transformer.from_crs(dsm_crs, 'EPSG:4326', always_xy=True)
            dsm_minlon, dsm_minlat = tf.transform(dsm_bounds.left, dsm_bounds.bottom)
            dsm_maxlon, dsm_maxlat = tf.transform(dsm_bounds.right, dsm_bounds.top)
        else:
            dsm_minlon, dsm_minlat, dsm_maxlon, dsm_maxlat = (
                dsm_bounds.left,
                dsm_bounds.bottom,
                dsm_bounds.right,
                dsm_bounds.top,
            )

        if str(gml_gdf.crs) != 'EPSG:4326':
            gml_wgs84 = gml_gdf.to_crs('EPSG:4326').total_bounds
        else:
            gml_wgs84 = gml_bounds
        gml_minlon, gml_minlat, gml_maxlon, gml_maxlat = gml_wgs84

        # Center of the map
        center_lat = (dsm_minlat + dsm_maxlat) / 2
        center_lon = (dsm_minlon + dsm_maxlon) / 2

        # Create a simple Folium map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')

        # Show the coverage of both sources
        folium.Rectangle(
            bounds=[[dsm_minlat, dsm_minlon], [dsm_maxlat, dsm_maxlon]],
            color='red', weight=2, fill=True, fill_opacity=0.1, popup='DSM gebied (hoogte data)',
        ).add_to(m)
        folium.Rectangle(
            bounds=[[gml_minlat, gml_minlon], [gml_maxlat, gml_maxlon]],
            color='blue', weight=2, fill=True, fill_opacity=0.1, popup='Wegen/fietspaden gebied',
        ).add_to(m)

        # Add a draw-control that only allows drawing rectangles
        draw = plugins.Draw(
            export=False,
            draw_options={
                'polyline': False,
                'polygon': False,
                'circle': False,
                'marker': False,
                'circlemarker': False,
                'rectangle': {'shapeOptions': {'color': 'orange', 'weight': 3}},
            },
            edit_options={'edit': False},
        )
        draw.add_to(m)

        # Javascript to send the bbox directly to Gradio
        map_id = m.get_name()
        js = f"""
        <script>
        // Use a small timeout to ensure the map object is fully initialized
        setTimeout(function() {{
            // Reference to the Leaflet map created by Folium
            var map = window.{map_id};
            if (!map) {{
                console.error('Could not find map object!');
                return;
            }}
            
            // When a rectangle is drawn:
            map.on('draw:created', function(e) {{
                var layer = e.layer;
                // For a rectangle, we can get the bounds directly
                var b = layer.getBounds();
                var coords = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]
                    .map(v => v.toFixed(6)).join(',');

                // Send the coordinates to the parent Gradio page
                console.log('🗺️ Map selection done. Sending coordinates:', coords);
                window.parent.postMessage({{type: 'bbox_update', coordinates: coords}}, '*');

                // NEW: Directly update the bbox textbox in the parent (Gradio) page
                try {{
                    var bboxContainer = window.parent.document.querySelector('#bbox-input');
                    if (bboxContainer) {{
                        var bboxInputElement = bboxContainer.querySelector('textarea, input');
                        if (bboxInputElement) {{
                            bboxInputElement.value = coords;
                            // Trigger input event so Gradio registers the change
                            bboxInputElement.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            console.log('✅ Bounding box updated directly from map iframe.');
                        }} else {{
                            console.warn('⚠️ Found bbox container but no input element.');
                        }}
                    }} else {{
                        console.warn('⚠️ Could not find bbox container with id #bbox-input in parent document.');
                    }}
                }} catch (err) {{
                    console.error('❌ Error while directly updating bbox in parent:', err);
                }}

                // Optional: copy to clipboard as a fallback
                if (navigator.clipboard) {{ 
                    navigator.clipboard.writeText(coords)
                        .then(() => console.log('📋 Coordinates also copied to clipboard.'))
                        .catch(err => console.error('Could not copy to clipboard:', err));
                }}
            }});
        }}, 250); // 250ms delay
        </script>
        """
        m.get_root().html.add_child(folium.Element(js))

        # Return the HTML which can be placed directly into Gradio
        return m._repr_html_()

    except Exception as e:
        # Simple error message if the map cannot be loaded
        return (
            f"<div style='padding:20px; border:2px solid red; border-radius:6px;'>"
            f"❌ Kaart kon niet worden geladen: {e}.<br/>"
            f"Voer de bounding-box handmatig in als min_lon,min_lat,max_lon,max_lat." 
            f"</div>"
        )

if __name__ == "__main__":
    app = create_complete_app()
    app.launch(server_port=7865, share=False) 
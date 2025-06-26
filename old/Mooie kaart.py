import rasterio
import rasterio.windows
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
import pvlib
import pandas as pd
from datetime import datetime, date, time, timedelta
import pytz
from math import tan, radians, cos, sin, sqrt
from numba import jit
import geopandas as gpd
from rasterio import features
from rasterio.mask import mask # Nieuwe import voor maskeren

# --- Configuratie ---
dsm_path = '2023_R_25GN1.TIF'
gml_path = 'bgt_wegdeel.gml' 

# Locatie voor de zonnebaanberekening
latitude = 52.37
longitude = 4.89
timezone = 'Europe/Amsterdam'

# Datum voor de analyse (standaard de huidige datum)
analysis_date = datetime.now() 

# Tijdstap voor de analyse van de schaduw (in minuten)
time_step_minutes = 30

# Start- en einduur voor de analyse
start_hour = 12
end_hour = 18

# Percentage van de DSM-data om te gebruiken voor de subset (voor snellere verwerking)
subset_width_percentage = 10
subset_height_percentage = 10

# Instellingen voor visualisatie
basemap_provider = ctx.providers.OpenStreetMap.Mapnik # Basemap provider

# --- Functie voor Schaduwberekening ---
@jit(nopython=True)
def calculate_shadows(dsm_data, nodata_value, transform, sun_azimuth_rad, sun_elevation_rad):
    """
    Berekent de schaduwen op basis van een Digital Surface Model (DSM), zonpositie en transformatiegegevens.

    Parameters:
    - dsm_data (np.array): De genormaliseerde DSM-data.
    - nodata_value (float): De NoData-waarde in de DSM (gebruikt voor NaN-controle).
    - transform (rasterio.transform.Affine): Affine transformatie van de DSM-subset.
    - sun_azimuth_rad (float): Zon-azimut in radialen.
    - sun_elevation_rad (float): Zon-elevatie in radialen.

    Returns:
    - np.array: Een schaduwmasker (0.0 voor schaduw, 1.0 voor zon, NaN voor NoData-gebieden).
    """
    rows, cols = dsm_data.shape
    shadow_mask = np.full(dsm_data.shape, 1.0, dtype=np.float32)
    pixel_res = abs(transform[0]) # Resolutie van een pixel in meters
    dx = sin(sun_azimuth_rad)
    dy = -cos(sun_azimuth_rad)
    tan_elevation = tan(sun_elevation_rad)
    # Maximale afstand in pixels om te controleren op schaduw (bijv. 200 meter)
    max_shadow_dist_pixels = int(200 / pixel_res) if pixel_res > 0 else 100

    for r in range(rows):
        for c in range(cols):
            h_pixel = dsm_data[r, c]
            if np.isnan(h_pixel):
                shadow_mask[r, c] = np.nan
                continue
            if tan_elevation <= 0:
                shadow_mask[r, c] = 0.0
                continue
            
            # Loop langs de zonnestraal om te controleren op blokkades
            # Gebruik een kleinere stap voor nauwkeurigheid, of pas 'max_shadow_dist_pixels' aan
            for i in range(1, max_shadow_dist_pixels + 1):
                check_c = c + int(round(i * dx)) # Directe stap op basis van dx, dy
                check_r = r + int(round(i * dy))

                if not (0 <= check_r < rows and 0 <= check_c < cols):
                    break
                
                dist_pixels = sqrt((check_c - c)**2 + (check_r - r)**2)
                dist_meters = dist_pixels * pixel_res
                min_blocker_height = h_pixel + dist_meters * tan_elevation
                
                h_check = dsm_data[check_r, check_c]
                if np.isnan(h_check):
                    continue
                
                if h_check > min_blocker_height:
                    shadow_mask[r, c] = 0.0
                    break
    return shadow_mask

# --- Hoofd Script ---
try:
    # --- 1. Voorbereidingen (DSM-data lezen en normaliseren) ---
    print("Voorbereiden van DSM-data...")
    with rasterio.open(dsm_path) as src:
        print(f"Originele DSM info: CRS={src.crs}, Shape={src.shape}")
        subset_rows = int(src.height * (subset_height_percentage / 100.0))
        subset_cols = int(src.width * (subset_width_percentage / 100.0))
        window = rasterio.windows.Window(0, 0, subset_cols, subset_rows)
        dsm_subset = src.read(1, window=window)
        subset_transform = src.window_transform(window)
        subset_bounds = rasterio.windows.bounds(window, src.transform)
        subset_extent = [subset_bounds[0], subset_bounds[2], subset_bounds[1], subset_bounds[3]]
        print(f"Subset gelezen: Shape={dsm_subset.shape}, Bounds=({subset_bounds[0]:.1f}, {subset_bounds[1]:.1f}, {subset_bounds[2]:.1f}, {subset_bounds[3]:.1f})")

        nodata_value = src.nodata if src.nodata is not None else -9999.0
        dsm_subset_float = dsm_subset.astype(np.float32)

        nodata_mask = (dsm_subset == nodata_value)

        dsm_subset_nan = dsm_subset_float.copy()
        dsm_subset_nan[nodata_mask] = np.nan

        dsm_for_shadow_calc = dsm_subset_float.copy()
        min_valid_height = np.nanmin(dsm_subset_nan)
        water_height = min_valid_height - 1.0 
        dsm_for_shadow_calc[nodata_mask] = water_height
        print(f"Minimale geldige landhoogte: {min_valid_height:.2f} m. Toegewezen waterhoogte: {water_height:.2f} m")

    dsm_normalized_for_shadow = dsm_for_shadow_calc - min_valid_height
    print(f"DSM genormaliseerd voor schaduw (min land = 0.0, water = {water_height - min_valid_height:.2f})")

    # --- 2. GML-data inlezen en voorbereiden (fietspaden en looppaden filteren) ---
    print("\nLaden en voorbereiden van weggedeelten data...")
    road_gdf = gpd.read_file(gml_path)
    print(f"Wegdelen gelezen: {len(road_gdf)} features, CRS={road_gdf.crs}")

    classificatie_kolom = 'function'
    fietspaden_waarden = ['fietspad'] 
    looppaden_waarden = ['voetpad'] 

    if classificatie_kolom in road_gdf.columns:
        filtered_gdf = road_gdf[
            (road_gdf[classificatie_kolom].isin(fietspaden_waarden)) |
            (road_gdf[classificatie_kolom].isin(looppaden_waarden))
        ]
        print(f"Gefilterd op fietspaden en looppaden. Aantal geselecteerde features: {len(filtered_gdf)}")
        if len(filtered_gdf) == 0:
            print("WAARSCHUWING: Geen fietspaden/looppaden gevonden met de opgegeven filterwaarden. Controleer de 'fietspaden_waarden' en 'looppaden_waarden'.")
            # Als er geen paden zijn om te visualiseren, stoppen we.
            raise ValueError("Geen fietspaden/looppaden gevonden om te analyseren.")
    else:
        print(f"FOUT: Kolom '{classificatie_kolom}' niet gevonden in het GML-bestand. Kan niet filteren op fietspaden/looppaden.")
        raise ValueError(f"Kolom '{classificatie_kolom}' niet gevonden in GML-bestand.")

    if filtered_gdf.crs != src.crs:
        print(f"Reprojecteren van gefilterde elementen van {filtered_gdf.crs} naar {src.crs}...")
        elements_gdf_reprojected = filtered_gdf.to_crs(src.crs)
    else:
        elements_gdf_reprojected = filtered_gdf

    # Maak een algemeen rastermasker van de gefilterde geometrieën voor de eerste plot
    print("Rasteriseren van alle gefilterde fietspaden en looppaden voor heatmap 1...")
    elements_mask = features.rasterize(
        ((geom, 1) for geom in elements_gdf_reprojected.geometry if geom is not None),
        out_shape=dsm_subset.shape,
        transform=subset_transform,
        fill=0,
        dtype='uint8'
    )
    print(f"Fietspaden/looppaden rastermasker gemaakt met vorm: {elements_mask.shape}")

    # --- 3. Tijdreeks Genereren ---
    local_tz = pytz.timezone(timezone)
    start_dt = local_tz.localize(datetime.combine(analysis_date, time(start_hour, 0)))
    end_dt = local_tz.localize(datetime.combine(analysis_date, time(end_hour, 0)))

    time_range = pd.date_range(start=start_dt, end=end_dt, freq=f'{time_step_minutes}min', tz=timezone)
    print(f"\nAnalyseren van schaduw voor {len(time_range)} tijdstippen ({start_dt.strftime('%H:%M')} tot {end_dt.strftime('%H:%M')})")

    # --- 4. Schaduw Accumulatie ---
    print("\nBerekenen van schaduw accumulatie...")
    shadow_accumulation = np.zeros_like(dsm_subset_nan, dtype=np.float32)

    for i, current_dt_local in enumerate(time_range):
        # print(f"Verwerken tijdstip {i+1}/{len(time_range)}: {current_dt_local.strftime('%H:%M')}") # Dit kan veel output geven
        current_dt_utc = current_dt_local.astimezone(pytz.utc)
        loc = pvlib.location.Location(latitude, longitude, tz='UTC')
        solar_pos = loc.get_solarposition(pd.to_datetime([current_dt_utc]))
        sun_azimuth_rad = radians(solar_pos['azimuth'].iloc[0])
        sun_elevation_rad = radians(solar_pos['apparent_elevation'].iloc[0])

        shadow_result = calculate_shadows(dsm_normalized_for_shadow, np.nan, subset_transform, sun_azimuth_rad, sun_elevation_rad)
        shadow_accumulation += shadow_result

    # shadow_percentage: 0 = 0% schaduw (volledige zon), 1 = 100% schaduw (volledige schaduw)
    shadow_percentage = 1 - (shadow_accumulation / len(time_range))
    print("Schaduw accumulatie berekening voltooid.")

    # --- 5. Eerste Heatmap Visualisatie (met gefilterd masker) ---
    print("\nMaken van eerste heatmap visualisatie (pixel-niveau) met gefilterde fietspaden en looppaden...")

    shadow_percentage_on_elements = shadow_percentage.copy()
    shadow_percentage_on_elements[elements_mask == 0] = np.nan # Zet pixels die GEEN fietspad of looppad zijn op NaN

    plt.figure(figsize=(12, 10))
    ax1 = plt.gca()
    ax1.set_xlim(subset_bounds[0], subset_bounds[2])
    ax1.set_ylim(subset_bounds[1], subset_bounds[3])
    ax1.set_axis_off()
    ax1.set_facecolor('white')

    ctx.add_basemap(ax1, crs=src.crs.to_string(), source=basemap_provider, zorder=1)

    # Plot de schaduwpercentage heatmap
    heatmap1 = ax1.imshow(shadow_percentage_on_elements,
                         cmap='RdYlBu', # Lage waarden (minder schaduw/meer zon) = Rood. Hoge waarden (meer schaduw) = Blauw.
                         extent=subset_extent,
                         alpha=0.8, 
                         vmin=0, vmax=1,
                         zorder=5
                         )

    cbar1 = plt.colorbar(heatmap1, ax=ax1, shrink=0.6, label='Percentage van tijd in schaduw (0=zon, 1=schaduw)')

    plt.title(f'Schaduw Accumulatie op Fietspaden en Looppaden (Pixel-niveau) - {analysis_date.strftime("%Y-%m-%d")}\n'
              f'Van {start_dt.strftime("%H:%M")} tot {end_dt.strftime("%H:%M")}')

    heatmap_filename1 = f'schaduw_heatmap_fietspaden_looppaden_pixel_{analysis_date.strftime("%Y%m%d")}.png'
    plt.savefig(heatmap_filename1, dpi=300, bbox_inches='tight')
    print(f"Eerste heatmap opgeslagen als: {heatmap_filename1}")
    plt.show()

    # --- 6. Bereken Gemiddelde Schaduw per Uniek Padsegment ---
    print("\nBerekenen van gemiddelde schaduw per uniek padsegment...")
    
    # Maak een kopie om de gemiddelde waarden in op te slaan
    elements_with_avg_shadow = elements_gdf_reprojected.copy()
    elements_with_avg_shadow['avg_shadow_percent'] = np.nan # Initialiseer nieuwe kolom

    # Transformeer het shadow_percentage raster naar een rasterio dataset-achtige structuur voor mask
    # Dit is nodig omdat rasterio.mask.mask een dataset object verwacht
    # We kunnen een in-memory dataset maken van de numpy array en transformatie
    from rasterio.io import MemoryFile
    
    # We slaan de shadow_percentage tijdelijk op als een in-memory TIF
    # Dit is efficiënter voor `rasterio.mask.mask` dan het handmatig herhaaldelijk rasterizen van geometrieën.
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=shadow_percentage.shape[0],
            width=shadow_percentage.shape[1],
            count=1,
            dtype=shadow_percentage.dtype,
            crs=src.crs,
            transform=subset_transform,
            nodata=np.nan # Gebruik NaN als NoData waarde in de tussentijdse raster
        ) as temp_dataset:
            temp_dataset.write(shadow_percentage, 1)

            # Loop door elke geometrie en bereken de gemiddelde schaduw
            for index, row in elements_with_avg_shadow.iterrows():
                geom = [row.geometry] # Mask verwacht een lijst van geometrieën
                
                if geom[0] is None or geom[0].is_empty:
                    continue # Sla lege of None geometrieën over

                try:
                    # Snijd het shadow_percentage raster uit met de huidige geometrie
                    out_image, out_transform = mask(temp_dataset, geom, crop=True, nodata=np.nan)
                    
                    # De gemaakte mask kan meerdere dimensies hebben, pak de eerste band
                    masked_data = out_image[0] 

                    # Bereken het gemiddelde, negeer NaNs
                    avg_val = np.nanmean(masked_data)
                    
                    if not np.isnan(avg_val):
                        # Converteer naar percentage (0-100) en sla op
                        elements_with_avg_shadow.loc[index, 'avg_shadow_percent'] = avg_val * 100
                except Exception as e:
                    #print(f"Fout bij maskeren of gemiddelde berekenen voor index {index}: {e}")
                    # Kan voorkomen als geometrie buiten het raster valt
                    elements_with_avg_shadow.loc[index, 'avg_shadow_percent'] = np.nan 

    print("Gemiddelde schaduw per padsegment berekend.")

    # --- 7. Tweede Heatmap Visualisatie (Padsegment-niveau) ---
    print("\nMaken van tweede heatmap visualisatie (padsegment-niveau)...")

    plt.figure(figsize=(12, 10))
    ax2 = plt.gca()
    ax2.set_xlim(subset_bounds[0], subset_bounds[2])
    ax2.set_ylim(subset_bounds[1], subset_bounds[3])
    ax2.set_axis_off()
    ax2.set_facecolor('white')

    ctx.add_basemap(ax2, crs=src.crs.to_string(), source=basemap_provider, zorder=1)

    # Plot de GeoDataFrame, gekleurd door de nieuwe 'avg_shadow_percent' kolom
    # cmap='RdYlBu' zorgt voor: lage waarden (0 = weinig schaduw) = Rood, hoge waarden (100 = veel schaduw) = Blauw
    plot2 = elements_with_avg_shadow.plot(
        ax=ax2,
        column='avg_shadow_percent',
        cmap='RdYlBu', # Rood voor weinig schaduw (warm), Blauw voor veel schaduw (koel)
        legend=True,
        legend_kwds={'label': "Gemiddeld percentage van tijd in schaduw (0=0%, 100=100%)", 'shrink': 0.6},
        vmin=0, vmax=100, # Vanaf 0% tot 100% schaduw
        linewidth=1.0,
        edgecolor='black', # Zwarte randen voor de paden voor betere zichtbaarheid
        zorder=5
    )

    plt.title(f'Gemiddelde Schaduw per Fietspad/Looppad ({analysis_date.strftime("%Y-%m-%d")})\n'
              f'Van {start_dt.strftime("%H:%M")} tot {end_dt.strftime("%H:%M")}')

    heatmap_filename2 = f'schaduw_heatmap_fietspaden_looppaden_gemiddeld_{analysis_date.strftime("%Y%m%d")}.png'
    plt.savefig(heatmap_filename2, dpi=300, bbox_inches='tight')
    print(f"Tweede heatmap opgeslagen als: {heatmap_filename2}")

    plt.show()

except ImportError as e:
    print(f"Fout: Benodigde bibliotheek niet gevonden. Zorg ervoor dat 'geopandas', 'fiona', 'rasterio' en 'pvlib' zijn geïnstalleerd. {e}")
except FileNotFoundError as e:
    print(f"Fout: Bestand niet gevonden op locatie: {e}. Controleer of '2023_R_25GN1.TIF' en 'bgt_wegdeel.gml' in dezelfde map staan als het script, of geef het volledige pad op.")
except ValueError as e:
    print(f"Fout in dataverwerking: {e}")
except Exception as e:
    print(f"Een onverwachte fout is opgetreden: {e}")
    import traceback
    traceback.print_exc()


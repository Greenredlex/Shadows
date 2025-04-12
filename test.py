import rasterio
import rasterio.windows
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
import pvlib
import pandas as pd
from datetime import datetime
import pytz # Voor tijdzone-bewuste datetime objecten
from math import tan, radians, cos, sin, sqrt
from numba import jit # Voor het versnellen van de schaduwberekening

# --- Configuratie ---
dsm_path = '2023_R_25GZ1.TIF' 

# Locatie (bij benadering voor Amsterdam centrum)
latitude = 52.37  # Graden Noord
longitude = 4.89   # Graden Oost
timezone = 'Europe/Amsterdam'

# Datum en tijd voor schaduwberekening (pas aan naar wens)
shadow_datetime_str = '2025-07-21 09:00:00' 

# Percentage van data te gebruiken (linksboven hoek)
subset_percentage = 10 # Gebruik 10% van rijen en kolommen

# DSM visualisatie
dsm_colormap = 'terrain' # Colormap voor hoogte
dsm_alpha = 1.0          # Volledig dekkend

# Schaduw visualisatie
shadow_colormap = 'Greys_r' # Zwart voor schaduw, wit voor zon
shadow_alpha = 0.6       # Transparantie van de schaduwlaag (zoals gevraagd)

# Basiskaart
basemap_provider = ctx.providers.CartoDB.Positron 
# --- Einde Configuratie ---

# --- Functie voor Schaduwberekening (versneld met Numba) ---
@jit(nopython=True) 
def calculate_shadows(dsm_data, nodata_value, transform, sun_azimuth_rad, sun_elevation_rad):
    rows, cols = dsm_data.shape
    shadow_mask = np.full(dsm_data.shape, 1.0, dtype=np.float32) 
    pixel_res = abs(transform[0]) 
    dx = sin(sun_azimuth_rad)
    dy = -cos(sun_azimuth_rad) 
    tan_elevation = tan(sun_elevation_rad)
    max_shadow_dist_pixels = int(200 / pixel_res) if pixel_res > 0 else 100 

    for r in range(rows):
        for c in range(cols):
            h_pixel = dsm_data[r, c]
            if h_pixel == nodata_value or np.isnan(h_pixel):
                shadow_mask[r, c] = np.nan # Gebruik NaN voor NoData in output
                continue
            if tan_elevation <= 0:
                 shadow_mask[r, c] = 0.0 
                 continue

            step = 1.0 
            for i in range(1, max_shadow_dist_pixels + 1):
                check_c = c + int(round(i * step * dx))
                check_r = r + int(round(i * step * dy))

                if not (0 <= check_r < rows and 0 <= check_c < cols):
                    break 
                
                dist_pixels = sqrt((check_c - c)**2 + (check_r - r)**2)
                dist_meters = dist_pixels * pixel_res # Werkelijke afstand
                
                # Hoogte van de zonnestraal OP DE LOCATIE van de check_pixel,
                # relatief aan de hoogte van de originele pixel
                # ray_height_at_check = h_pixel + dist_meters * tan_elevation 
                # Correctie: Hoogte van de zonnestraal die de originele pixel (r,c) zou raken,
                # maar dan op de horizontale positie van de check pixel.
                # Of andersom: de minimale hoogte die de check_pixel moet hebben om de originele pixel te beschaduwen.
                min_blocker_height = h_pixel + dist_meters * tan_elevation

                h_check = dsm_data[check_r, check_c]
                if np.isnan(h_check): # Gebruik nu np.isnan omdat we NoData naar NaN hebben gezet
                    continue 

                if h_check > min_blocker_height:
                    shadow_mask[r, c] = 0.0 # Schaduw
                    break 
    return shadow_mask

# --- Hoofd Script ---
try:
    # --- 1. Zonpositie Berekenen ---
    local_tz = pytz.timezone(timezone)
    dt_naive = datetime.strptime(shadow_datetime_str, '%Y-%m-%d %H:%M:%S')
    dt_aware = local_tz.localize(dt_naive)
    dt_utc = dt_aware.astimezone(pytz.utc)
    time = pd.to_datetime([dt_utc])
    loc = pvlib.location.Location(latitude, longitude, tz='UTC') 
    solar_pos = loc.get_solarposition(time)
    sun_azimuth_deg = solar_pos['azimuth'].iloc[0]
    sun_elevation_deg = solar_pos['apparent_elevation'].iloc[0]
    print(f"Zonpositie voor {dt_aware.strftime('%Y-%m-%d %H:%M:%S %Z%z')}:")
    print(f"  Azimut: {sun_azimuth_deg:.2f} graden (0=N, 90=E, 180=S, 270=W)")
    print(f"  Elevatie: {sun_elevation_deg:.2f} graden (boven horizon)")
    sun_azimuth_rad = radians(sun_azimuth_deg)
    sun_elevation_rad = radians(sun_elevation_deg)

    # --- 2. DSM Data Lezen (Subset) ---
    with rasterio.open(dsm_path) as src:
        print(f"\nOriginele DSM info:")
        print(f"  CRS: {src.crs}")
        print(f"  Shape: {src.shape}")
        
        subset_rows = int(src.height * (subset_percentage / 100.0))
        subset_cols = int(src.width * (subset_percentage / 100.0))
        window = rasterio.windows.Window(0, 0, subset_cols, subset_rows)
        print(f"\nLezen van subset:")
        print(f"  Window: {window}")

        dsm_subset = src.read(1, window=window)
        subset_transform = src.window_transform(window)
        subset_bounds = rasterio.windows.bounds(window, src.transform)
        # Correctie indexering: [left, right, bottom, top]
        subset_extent = [subset_bounds[0], subset_bounds[2], subset_bounds[1], subset_bounds[3]] 
        print(f"  Subset Shape: {dsm_subset.shape}")
        print(f"  Subset Transform: {subset_transform}")
        print(f"  Subset Bounds: ({subset_bounds[0]:.2f}, {subset_bounds[1]:.2f}, {subset_bounds[2]:.2f}, {subset_bounds[3]:.2f})")

        nodata_value = src.nodata if src.nodata is not None else -9999.0 
        dsm_subset = dsm_subset.astype(np.float32) 
        nodata_mask = (dsm_subset == nodata_value)
        dsm_subset_nan = dsm_subset.copy() # Maak kopie voor schaduwberekening
        dsm_subset_nan[nodata_mask] = np.nan # Gebruik NaN voor berekeningen
        
        # Behoud originele NoData waarde voor plotten DSM (optioneel, maar kan helpen bij kleurbalk)
        # dsm_subset[nodata_mask] = np.nan # Of zet ook hier NaN als je dat prefereert

    # --- 3. Schaduw Berekenen ---
    print("\nStarten met schaduwberekening (kan even duren)...")
    # Geef de NaN-versie en de *numerieke* NoData waarde mee aan de functie
    # De functie verwacht nu NaN als input NoData, maar we gebruiken np.nan intern
    shadow_result = calculate_shadows(dsm_subset_nan, np.nan, subset_transform, sun_azimuth_rad, sun_elevation_rad)
    print("Schaduwberekening voltooid.")

    # Maskeer NoData gebieden in het schaduwresultaat voor plotten
    shadow_result[np.isnan(dsm_subset_nan)] = np.nan # Zet NoData terug naar NaN

    # --- 4. Plotten ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # --- NIEUW: Stel de assen limieten expliciet in ---
    # Gebruik de berekende grenzen van de subset (in RD New coordinaten)
    ax.set_xlim(subset_bounds[0], subset_bounds[2]) # xlim = (left, right)
    ax.set_ylim(subset_bounds[1], subset_bounds[3]) # ylim = (bottom, top)
    # --- Einde NIEUW ---

    # --- Laag 1: Basiskaart ---
    print("Basiskaart toevoegen...")
    # Contextily zou nu de correcte grenzen moeten gebruiken
    ctx.add_basemap(ax, crs=src.crs.to_string(), source=basemap_provider, zorder=1) 

    # --- Laag 2: DSM Hoogtekaart ---
    # (Rest van de plot code blijft hetzelfde)
    print("DSM plotten...")
    min_h = np.nanmin(dsm_subset_nan)
    max_h = np.nanmax(dsm_subset_nan)
    img_dsm = ax.imshow(dsm_subset_nan, 
                        cmap=dsm_colormap, 
                        extent=subset_extent, 
                        alpha=dsm_alpha,
                        vmin=min_h,      
                        vmax=max_h,
                        zorder=5)        

    # --- Laag 3: Schaduwkaart ---
    print("Schaduwlaag plotten...")
    img_shadow = ax.imshow(shadow_result, 
                        cmap=shadow_colormap, 
                        extent=subset_extent, 
                        alpha=shadow_alpha, 
                        vmin=0, 
                        vmax=1, 
                        zorder=10)       

    # --- Kleurenbalk voor DSM ---
    cbar = fig.colorbar(img_dsm, ax=ax, shrink=0.6, label='Hoogte (m)') 
        
    # --- Legenda voor Schaduw ---
    from matplotlib.lines import Line2D
    shadow_patch = Line2D([0], [0], marker='s', color='w', 
                        markerfacecolor='black', markersize=10, alpha=shadow_alpha, 
                        label='Schaduw', linestyle='') 
    ax.legend(handles=[shadow_patch], loc='lower right', title="Legenda")

    # Zet assen uit (kan eventueel na add_basemap, maar vaak beter aan het eind)
    ax.set_axis_off()

    # Titel en informatie
    ax.set_title(f"Schaduw & Hoogte Amsterdam (Subset {subset_percentage}%) - {dt_aware.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                f"Zon Azimut: {sun_azimuth_deg:.1f}°, Elevatie: {sun_elevation_deg:.1f}°")

    plt.tight_layout()
    plt.show()
    print("Plot voltooid.")

    # (Rest van de try...except blokken)


except ImportError as e:
    print(f"Fout: Benodigde bibliotheek niet gevonden. {e}")
    print("Installeer de vereiste bibliotheken: pip install rasterio matplotlib contextily numpy pvlib-python pytz numba")
except rasterio.RasterioIOError as e:
    print(f"Fout bij het openen of lezen van het DSM-bestand: {e}")
except FileNotFoundError:
    print(f"Fout: Bestand niet gevonden op locatie: {dsm_path}")
except Exception as e:
    print(f"Een onverwachte fout is opgetreden: {e}")
    import traceback
    traceback.print_exc() 


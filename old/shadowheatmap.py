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

# --- Configuratie ---
dsm_path = '2023_R_25GZ1.TIF' 

# Locatie 
latitude = 52.37  
longitude = 4.89   
timezone = 'Europe/Amsterdam'

# Datum voor de analyse
analysis_date = datetime.now() # Kies de dag

# Tijdstap voor analyse
time_step_minutes = 5

# Start en eindtijd
start_hour = 5 
end_hour = 21 

# Percentage van data te gebruiken
subset_width_percentage = 10
subset_height_percentage = 10 

# Visualisatie instellingen
dsm_colormap = 'terrain' 
dsm_alpha = 1          
shadow_colormap = 'Greys_r' 
shadow_alpha = 0.6       
basemap_provider = ctx.providers.OpenStreetMap.Mapnik

# --- Functie voor Schaduwberekening ---
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
            if np.isnan(h_pixel): 
                shadow_mask[r, c] = np.nan 
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
    # --- 1. Voorbereidingen (Data lezen, Normaliseren) ---
    print("Voorbereiden van data...")
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

        # Create mask for NoData areas (water)
        nodata_mask = (dsm_subset == nodata_value)

        # Create DSM for visualization (with NaNs for NoData)
        dsm_subset_nan = dsm_subset_float.copy()
        dsm_subset_nan[nodata_mask] = np.nan

        # Create DSM for shadow calculation (replace NoData with flat water level)
        dsm_for_shadow_calc = dsm_subset_float.copy()
        min_valid_height = np.nanmin(dsm_subset_nan)
        water_height = min_valid_height - 1.0
        dsm_for_shadow_calc[nodata_mask] = water_height
        print(f"Min valid land height: {min_valid_height:.2f} m. Assigned water height: {water_height:.2f} m")

    # Normalize heights for shadow calculation
    dsm_normalized_for_shadow = dsm_for_shadow_calc - min_valid_height
    print(f"DSM genormaliseerd voor schaduw (min land = 0.0, water = {water_height - min_valid_height:.2f})")

    # --- 2. Tijdreeks Genereren ---
    local_tz = pytz.timezone(timezone)
    start_dt = local_tz.localize(datetime.combine(analysis_date, time(start_hour, 0)))
    end_dt = local_tz.localize(datetime.combine(analysis_date, time(end_hour, 0)))
    
    # Genereer tijdstippen met pandas
    time_range = pd.date_range(start=start_dt, end=end_dt, freq=f'{time_step_minutes}min', tz=timezone)
    print(f"\nAnalyseren van schaduw voor {len(time_range)} tijdstippen ({start_dt.strftime('%H:%M')} tot {end_dt.strftime('%H:%M')})")

    # --- 3. Schaduw Accumulatie ---
    print("\nBerekenen van schaduw accumulatie...")
    
    # Initialize shadow accumulation array
    shadow_accumulation = np.zeros_like(dsm_subset_nan, dtype=np.float32)
    
    # Process each time step
    for i, current_dt_local in enumerate(time_range):
        print(f"Verwerken tijdstip {i+1}/{len(time_range)}: {current_dt_local.strftime('%H:%M')}")
        
        current_dt_utc = current_dt_local.astimezone(pytz.utc)
        loc = pvlib.location.Location(latitude, longitude, tz='UTC')
        solar_pos = loc.get_solarposition(pd.to_datetime([current_dt_utc]))
        sun_azimuth_rad = radians(solar_pos['azimuth'].iloc[0])
        sun_elevation_rad = radians(solar_pos['apparent_elevation'].iloc[0])
        
        # Calculate shadow for this time step
        shadow_result = calculate_shadows(dsm_normalized_for_shadow, np.nan, subset_transform, sun_azimuth_rad, sun_elevation_rad)
        
        # Accumulate shadows (0 for shadow, 1 for sun)
        shadow_accumulation += shadow_result

    # Normalize to get percentage of time in shadow
    shadow_percentage = 1 - (shadow_accumulation / len(time_range))

    # --- 4. Heatmap Visualisatie ---
    print("\nMaken van heatmap visualisatie...")
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    ax.set_xlim(subset_bounds[0], subset_bounds[2])
    ax.set_ylim(subset_bounds[1], subset_bounds[3])
    ax.set_axis_off()
    
    # Add basemap
    ctx.add_basemap(ax, crs=src.crs.to_string(), source=basemap_provider, zorder=1)
    
    # Plot shadow percentage heatmap
    heatmap = ax.imshow(shadow_percentage, 
                       cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (red = more shadow)
                       extent=subset_extent,
                       alpha=0.7,
                       vmin=0, vmax=1,
                       zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.6, label='Percentage van tijd in schaduw')
    
    # Add title
    plt.title(f'Schaduw Accumulatie - {analysis_date.strftime("%Y-%m-%d")}\n'
              f'Van {start_dt.strftime("%H:%M")} tot {end_dt.strftime("%H:%M")}')
    
    # Save the heatmap
    heatmap_filename = f'schaduw_heatmap_{analysis_date.strftime("%Y%m%d")}.png'
    plt.savefig(heatmap_filename, dpi=300, bbox_inches='tight')
    print(f"Heatmap opgeslagen als: {heatmap_filename}")
    
    # Show the heatmap
    plt.show()

except ImportError as e:
    print(f"Fout: Benodigde bibliotheek niet gevonden. {e}")
except FileNotFoundError as e:
    print(f"Fout: Bestand niet gevonden op locatie: {dsm_path}")
except Exception as e:
    print(f"Een onverwachte fout is opgetreden: {e}")
    import traceback
    traceback.print_exc()

import rasterio
import rasterio.windows
import matplotlib.pyplot as plt
import matplotlib.animation as animation # Nieuwe import
import numpy as np
import contextily as ctx
import pvlib
import pandas as pd
from datetime import datetime, date, time, timedelta # Meer datetime tools
import pytz 
from math import tan, radians, cos, sin, sqrt
from numba import jit 

# --- Configuratie ---
dsm_path = '2023_R_25GZ1.TIF' 

# Locatie 
latitude = 52.37  
longitude = 4.89   
timezone = 'Europe/Amsterdam'

# Datum voor de animatie
animation_date = date(2025, 7, 21) # Kies de dag

# Tijdstap voor animatie
time_step_minutes = 10

# Start en eindtijd (optioneel, anders hele dag)
# Als None, pakt van zonsopgang tot zonsondergang (ongeveer)
start_hour = 5 
end_hour = 21 

# Percentage van data te gebruiken
subset_width_percentage = 10
subset_height_percentage = 10 

# Visualisatie instellingen (zoals de laatste werkende versie)
dsm_colormap = 'terrain' 
dsm_alpha = 1 # 1.0          
shadow_colormap = 'Greys_r' # Schaduw(0)=Licht, Zon(1)=Donker
shadow_alpha = 0.6       
basemap_provider = ctx.providers.OpenStreetMap.Mapnik #ctx.providers.CartoDB.Positron

# Animatie snelheid (milliseconden tussen frames)
animation_interval = 500 # ms (0.5 seconde per frame)
# --- Einde Configuratie ---

# --- Functie voor Schaduwberekening (ongewijzigd) ---
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
    # Dit gebeurt nu maar één keer, voor de animatie start
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
        dsm_subset_float = dsm_subset.astype(np.float32) # Use float for calculations

        # Create mask for NoData areas (water)
        nodata_mask = (dsm_subset == nodata_value)

        # Create DSM for visualization (with NaNs for NoData)
        dsm_subset_nan = dsm_subset_float.copy()
        dsm_subset_nan[nodata_mask] = np.nan

        # Create DSM for shadow calculation (replace NoData with flat water level)
        dsm_for_shadow_calc = dsm_subset_float.copy()
        min_valid_height = np.nanmin(dsm_subset_nan) # Min height of actual land/buildings
        water_height = min_valid_height - 1.0 # Set water level 1m below lowest land
        dsm_for_shadow_calc[nodata_mask] = water_height
        print(f"Min valid land height: {min_valid_height:.2f} m. Assigned water height: {water_height:.2f} m")

    # Normalize heights for shadow calculation (based on the modified DSM)
    # Water will have a negative normalized height (e.g., -1.0)
    dsm_normalized_for_shadow = dsm_for_shadow_calc - min_valid_height
    print(f"DSM genormaliseerd voor schaduw (min land = 0.0, water = {water_height - min_valid_height:.2f})")

    # --- 2. Tijdreeks Genereren ---
    local_tz = pytz.timezone(timezone)
    start_dt = local_tz.localize(datetime.combine(animation_date, time(start_hour, 0)))
    end_dt = local_tz.localize(datetime.combine(animation_date, time(end_hour, 0)))
    
    # Genereer tijdstippen met pandas
    time_range = pd.date_range(start=start_dt, end=end_dt, freq=f'{time_step_minutes}min', tz=timezone)
    print(f"\nGenereren van animatie voor {len(time_range)} frames ({start_dt.strftime('%H:%M')} tot {end_dt.strftime('%H:%M')})")

    # --- 3. Plot Initialisatie ---
    print("Initialiseren van plot...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(subset_bounds[0], subset_bounds[2]) 
    ax.set_ylim(subset_bounds[1], subset_bounds[3]) 
    ax.set_axis_off() # Assen meteen uitzetten

    # Laag 1: Basiskaart (één keer tekenen)
    ctx.add_basemap(ax, crs=src.crs.to_string(), source=basemap_provider, zorder=1) 

    # Laag 2: DSM Hoogtekaart (één keer tekenen)
    min_h = np.nanmin(dsm_subset_nan) 
    max_h = np.nanmax(dsm_subset_nan)
    img_dsm = ax.imshow(dsm_subset_nan, 
                        cmap=dsm_colormap, 
                        extent=subset_extent, 
                        alpha=dsm_alpha, 
                        vmin=min_h, vmax=max_h,
                        zorder=5)        
    cbar = fig.colorbar(img_dsm, ax=ax, shrink=0.6, label='Hoogte (m tov NAP)') 

    # Laag 3: Schaduwkaart (initialiseren met lege data of eerste frame)
    # We berekenen het eerste frame hier alvast voor een startbeeld
    first_dt_utc = time_range[0].astimezone(pytz.utc)
    loc = pvlib.location.Location(latitude, longitude, tz='UTC') 
    first_solar_pos = loc.get_solarposition(pd.to_datetime([first_dt_utc]))
    first_sun_az_rad = radians(first_solar_pos['azimuth'].iloc[0])
    first_sun_el_rad = radians(first_solar_pos['apparent_elevation'].iloc[0])
    
    initial_shadow = calculate_shadows(dsm_normalized_for_shadow, np.nan, subset_transform, first_sun_az_rad, first_sun_el_rad)
    # initial_shadow[np.isnan(dsm_subset_nan)] = np.nan # Don't mask shadows based on original NaNs

    img_shadow = ax.imshow(initial_shadow, # Start met eerste frame
                           cmap=shadow_colormap, 
                           extent=subset_extent, 
                           alpha=shadow_alpha, 
                           vmin=0, vmax=1, 
                           zorder=10)       

    # Titel object (om later te updaten)
    title_obj = ax.set_title("") # Start met lege titel

    # --- 4. Animatie Update Functie ---
    def update(frame):
        current_dt_local = time_range[frame]
        print(f"Frame {frame+1}/{len(time_range)}: {current_dt_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Bereken zonpositie
        current_dt_utc = current_dt_local.astimezone(pytz.utc)
        solar_pos = loc.get_solarposition(pd.to_datetime([current_dt_utc]))
        sun_azimuth_deg = solar_pos['azimuth'].iloc[0]
        sun_elevation_deg = solar_pos['apparent_elevation'].iloc[0]
        sun_azimuth_rad = radians(sun_azimuth_deg)
        sun_elevation_rad = radians(sun_elevation_deg)

        # Bereken schaduw
        shadow_result = calculate_shadows(dsm_normalized_for_shadow, np.nan, subset_transform, sun_azimuth_rad, sun_elevation_rad)
        # shadow_result[np.isnan(dsm_subset_nan)] = np.nan # Don't mask shadows based on original NaNs

        # Update schaduw laag data
        img_shadow.set_data(shadow_result)

        # Update titel
        title_str = (f"Schaduw Amsterdam - {current_dt_local.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                     f"Zon Azimut: {sun_azimuth_deg:.1f}°, Elevatie: {sun_elevation_deg:.1f}°")
        title_obj.set_text(title_str)

        # Return de geupdatete elementen (belangrijk voor blitting)
        return img_shadow, title_obj

    # --- 5. Animatie Creëren en Tonen/Opslaan ---
    print("\nCreëren van animatie object...")
    # blit=True kan sneller zijn, maar soms problemen geven. Zet op False als het raar doet.
    ani = animation.FuncAnimation(fig, update, frames=len(time_range), 
                                  interval=animation_interval, blit=True, repeat=False) 

    # Optie 1: Toon de animatie interactief
    # print("Tonen van animatie (sluit venster om door te gaan)...")
    # plt.show() 

    # Optie 2: Sla de animatie op (commentarieer plt.show() hierboven uit als je alleen wilt opslaan)
    # Zorg dat je 'imagemagick' (voor GIF) of 'ffmpeg' (voor MP4) hebt geïnstalleerd!
    output_filename_gif = f'schaduw_animatie_{animation_date.strftime("%Y%m%d")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.gif'
    # output_filename_mp4 = f'schaduw_animatie_{animation_date.strftime("%Y%m%d")}.mp4'
    print(f"\nOpslaan van animatie als GIF: {output_filename_gif} (kan lang duren!)...")
    ani.save(output_filename_gif, writer='imagemagick', fps=1000/animation_interval) 
    # print(f"Opslaan van animatie als MP4: {output_filename_mp4} (kan lang duren!)...")
    # ani.save(output_filename_mp4, writer='ffmpeg', fps=1000/animation_interval, dpi=150) # dpi voor resolutie

    print("\nScript voltooid.")

except ImportError as e:
    if 'animation' in str(e):
         print(f"Fout: Matplotlib animatie module niet gevonden? {e}")
    else:
         print(f"Fout: Benodigde bibliotheek niet gevonden. {e}")
except FileNotFoundError as e:
     if 'imagemagick' in str(e).lower() or 'ffmpeg' in str(e).lower():
         print(f"Fout: Kan animatie writer niet vinden: {e}")
         print("Zorg dat 'imagemagick' (voor GIF) of 'ffmpeg' (voor MP4) is geïnstalleerd en in het systeem PATH staat.")
     else:
        print(f"Fout: Bestand niet gevonden op locatie: {dsm_path}")
except Exception as e:
    print(f"Een onverwachte fout is opgetreden: {e}")
    import traceback
    traceback.print_exc()

🌞 Complete Shadow Analysis Tool — Quickstart


This project lets you analyze how much of the day different places are in shade, using a DSM (height map) and road data. It calculates pixel-level shadow percentages over a time range and summarizes shade on bike/footpath segments. You get clean maps, a sun-path plot, and optional GeoTIFF exports you can open in QGIS/ArcGIS.

Yeah, it’s pretty slick.


---

What it does (in plain English)

- Loads a DSM GeoTIFF (e.g., OUTPUT.TIF) and road data (BGT GML).

- Lets you pick an area on a mini-map (draw a rectangle).

- Simulates sun position over your chosen time range.

- Calculates where shadows fall per time step (using a fast Numba-optimized core).

- Produces:
	- DSM + detected “shadow-casting objects” overlay,

	- Sun path plot,

	- Pixel-level shadow percentage heatmap,

	- Average shadow per bike/footpath segment.


- Optionally saves everything as GeoTIFFs for GIS workflows.

You run it in a Gradio web UI. It logs progress live, so it feels pro.


---

Requirements


Files expected in the project folder:


- OUTPUT.TIF — DSM raster (required)

- bgt_wegdeel.gml — BGT road/foot/bike paths (required)

- split_bike_foot_paths.gpkg — Optional. Auto-generated or loaded if present

- road_segment_splitter.py — Optional. Improves per-road resolution

Python packages (auto-checked by the app):


- rasterio, geopandas, pvlib, folium, contextily, numba

- plus common stuff: numpy, pandas, matplotlib, shapely, pyproj, pillow, pytz

Tip: Use a virtual environment.


---

Install

	python -m venv .venv
	source .venv/bin/activate    # On Windows: .venv\Scripts\activate
	pip install -r requirements.txt

If you don’t have a requirements.txt, install these manually:


	pip install gradio numpy pandas rasterio geopandas matplotlib folium pvlib pillow contextily numba shapely pyproj pytz

Note: On some systems, installing geopandas/rasterio may require GDAL. Use your OS package manager or wheels.


---

Run

	python app.py

Then open the local URL printed by Gradio (usually http://127.0.0.1:7865).

If you renamed the file, make sure the main file still calls:


	if __name__ == "__main__":
	    app = create_complete_app()
	    app.launch(server_port=7865, share=False)


---

Using the app

1. 


Set your parameters


	- Date (YYYY-MM-DD)

	- Time step (e.g., 30 minutes)

	- Start/End hours

	- Object height threshold (meters): everything taller than this is treated as a shadow-casting object (trees/buildings).


2. 
Select the area


	- Draw a rectangle on the map.

	- The bounding box auto-fills below. You can edit it manually if needed.


3. 
Run the analysis


	- Click “Start Complete Analyse”.

	- Watch the live logs (you’ll see sun azimuth/elevation and progress).

	- View 4 outputs:
		- DSM + objects overlay

		- Sun path

		- Pixel-level shadow heatmap

		- Per-bike/footpath average shadow map



4. 
Optional: Save to GeoTIFF


	- Toggle “Save analysis data” if you want GIS-ready outputs in the data/ folder.



---

Outputs


When saving is enabled, you’ll get:


- data/dsm_with_objects_[timestamp].tif
	- 2 bands: normalized DSM, object mask


- data/shadow_percentage_[timestamp].tif
	- Pixel-level % time in shadow (0–100)


- data/road_shadows_[timestamp].tif
	- Rasterized per-segment shadow percentages


You can load these in QGIS/ArcGIS and style them your way.


---

Notes and pro tips

- Performance: The core shadow computation is Numba-jitted and respects raster resolution. Smaller areas and larger time steps run faster.

- Coordinate systems: The app handles CRS conversions automatically for the map and analysis.

- Better per-road results: If road_segment_splitter.py is available, the app uses pre-split 50m segments for cleaner per-segment stats.

- Fallbacks: If split segments aren’t found, it uses the original GML roads and clips to your area.


---

Troubleshooting

- Missing files: The top of the UI shows a “System Status” panel. Fix any ❌ items before running.

- Basemap issues: If a basemap fails to load, the app continues with analysis and logs a warning.

- GDAL/GEOS problems: Install via OS packages or use prebuilt wheels for rasterio/geopandas.


---

Why this is cool

- Real shadow simulation across time, not just static shade.

- Clean GIS exports for serious analysis.

- Simple web UI with a one-rectangle selection that “just works”.

- You look like you built a tiny solar lab in a weekend.

import geopandas as gpd
import os
import re
import json
import warnings

# --- Configuration ---
# !!! --- REQUIRED: SET THESE VALUES --- !!!
GML_FILE_PATH = 'bgt_wegdeel.gml'  # Replace with the path to YOUR GML file
ATTRIBUTE_COLUMN = 'function'     # The column name containing categories (e.g., 'voetpad')
OUTPUT_GEOJSON_DIR = 'geojson_layers_wgs84' # Name for the output folder

# !!! --- OPTIONAL: ADJUST AS NEEDED --- !!!
# If GeoPandas cannot detect the CRS, assume this one (e.g., for Dutch RD New)
ASSUMED_ORIGINAL_CRS = 'EPSG:28992'
# Enable/disable geometry simplification
ENABLE_SIMPLIFICATION = True
# Simplification tolerance (in units of the *original* CRS - e.g., meters for EPSG:28992)
# Smaller values = less simplification, more detail, larger files.
SIMPLIFICATION_TOLERANCE = 0.5
# Handle features with missing values in the attribute column?
FILL_NA_CATEGORY_WITH = 'Unknown' # Set to None to skip features with missing values

# --- Helper function to create safe filenames ---
def sanitize_filename(name):
    """Creates a filesystem-safe filename from a category name."""
    try:
        # Convert to string, strip whitespace, replace spaces
        name = str(name).strip().replace(' ', '_')
        # Remove characters problematic in filenames (allow letters, numbers, underscore, hyphen, dot)
        name = re.sub(r'(?u)[^-\w.]', '', name)
        # Prevent empty or dot-only filenames
        if not name or name.strip('.') == '':
            return f"layer_{abs(hash(name))}" # Use hash for a unique fallback
        # Limit length (optional)
        return f"layer_{name[:50]}" # Limit base name length
    except Exception as e:
        print(f"  Warning: Could not sanitize name '{name}'. Using fallback. Error: {e}")
        return f"layer_{abs(hash(name))}" # Use hash for a unique fallback

# --- Main Processing Function ---
def generate_geojson_layers(
    gml_path,
    attribute_col,
    output_dir,
    assumed_crs=None,
    simplify=True,
    tolerance=0.5,
    fill_na=None
):
    """
    Reads a GML file, processes features, and saves categorized GeoJSON layers
    in EPSG:4326 (WGS84).
    """
    print("-" * 60)
    print(f"Starting GeoJSON layer generation for: {gml_path}")
    print("-" * 60)

    # 1. Check if GML file exists
    if not os.path.exists(gml_path):
        print(f"!!! Error: Input GML file not found at '{gml_path}'")
        return False

    # 2. Create output directory
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        else:
            print(f"Output directory already exists: {output_dir}")
            # Consider adding logic here to clear the directory if needed:
            # print("Warning: Output directory exists. Files may be overwritten.")
            # import shutil
            # shutil.rmtree(output_dir)
            # os.makedirs(output_dir)
    except OSError as e:
        print(f"!!! Error: Could not create output directory '{output_dir}': {e}")
        return False

    # 3. Read GML file
    try:
        print(f"Reading GML file: {os.path.basename(gml_path)}...")
        # Ignore potential XML schema warnings during read if they occur
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=UserWarning)
             gdf = gpd.read_file(gml_path)
        print(f"-> Successfully read {len(gdf)} features.")
    except Exception as e:
        print(f"!!! Error: Failed to read GML file: {e}")
        print("   Possible causes: File corruption, unsupported GML structure, missing dependencies (GDAL/Fiona).")
        return False

    # 4. Check Attribute Column
    if attribute_col not in gdf.columns:
        print(f"!!! Error: Attribute column '{attribute_col}' not found in the GML.")
        print(f"   Available columns are: {list(gdf.columns)}")
        return False
    print(f"Using attribute column: '{attribute_col}'")

    # 5. Handle Original CRS
    original_crs = gdf.crs
    if original_crs:
        print(f"Detected original CRS: {original_crs.name} ({original_crs.to_string()})")
    elif assumed_crs:
        print(f"Warning: Original CRS not detected. Assuming '{assumed_crs}'.")
        try:
            gdf.set_crs(assumed_crs, inplace=True, allow_override=True)
            original_crs = gdf.crs # Update variable
            print(f"-> Set CRS to {original_crs.name}")
        except Exception as e:
            print(f"!!! Error: Failed to set assumed CRS '{assumed_crs}': {e}")
            print("   Cannot proceed reliably without a known starting CRS.")
            return False
    else:
        print("!!! Error: Original CRS not detected and no 'assumed_crs' provided.")
        print("   Cannot perform reprojection accurately.")
        return False

    # 6. Initial Geometry Cleaning
    print("Cleaning initial geometries...")
    initial_count = len(gdf)
    # Ensure geometry column exists and is active
    if 'geometry' not in gdf.columns or not hasattr(gdf, 'geometry'):
         print("!!! Error: Could not find geometry data in the GML.")
         return False
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.is_valid]
    cleaned_count = len(gdf)
    print(f"-> Removed {initial_count - cleaned_count} empty/invalid initial geometries.")
    if cleaned_count == 0:
        print("!!! Error: No valid geometries remaining after initial cleaning.")
        return False

    # 7. Simplification (Optional)
    if simplify:
        print(f"Simplifying geometries (Tolerance: {tolerance}, Units: {original_crs.axis_info[0].unit_name if original_crs.axis_info else 'unknown'})...")
        try:
            # Simplify requires a valid CRS and geometry
            gdf['geometry'] = gdf.geometry.simplify(tolerance)
            # Clean again after simplification
            simplified_count = len(gdf)
            gdf = gdf[~gdf.geometry.is_empty]
            gdf = gdf[gdf.geometry.is_valid]
            final_simplified_count = len(gdf)
            print(f"-> Removed {simplified_count - final_simplified_count} empty/invalid geometries post-simplification.")
            if final_simplified_count == 0:
                 print("!!! Error: No valid geometries remaining after simplification.")
                 return False
        except Exception as e:
            print(f"Warning: Error during simplification: {e}. Proceeding without simplification.")
    else:
        print("Skipping geometry simplification.")

    # 8. Reprojection to WGS84 (EPSG:4326)
    target_crs = 'EPSG:4326'
    print(f"Reprojecting data to {target_crs} (WGS84 Lat/Lon)...")
    try:
        gdf_wgs84 = gdf.to_crs(target_crs)
        print("-> Reprojection successful.")
    except Exception as e:
        print(f"!!! Error: Failed during reprojection to {target_crs}: {e}")
        return False

    # 9. Final Geometry Cleaning (Post-Reprojection)
    print("Cleaning geometries after reprojection...")
    initial_wgs_count = len(gdf_wgs84)
    gdf_wgs84 = gdf_wgs84[~gdf_wgs84.geometry.is_empty]
    gdf_wgs84 = gdf_wgs84[gdf_wgs84.geometry.is_valid]
    final_wgs_count = len(gdf_wgs84)
    print(f"-> Removed {initial_wgs_count - final_wgs_count} empty/invalid post-reprojection geometries.")
    if final_wgs_count == 0:
        print("!!! Error: No valid geometries remaining after final cleaning.")
        return False

    # 10. Handle Missing Attribute Values
    if fill_na is not None:
        print(f"Handling missing values in '{attribute_col}' (filling with '{fill_na}')...")
        original_na_count = gdf_wgs84[attribute_col].isna().sum()
        gdf_wgs84[attribute_col] = gdf_wgs84[attribute_col].fillna(fill_na)
        print(f"-> Filled {original_na_count} missing values.")
    else:
        print(f"Skipping features with missing values in '{attribute_col}'...")
        original_na_count = gdf_wgs84[attribute_col].isna().sum()
        gdf_wgs84 = gdf_wgs84.dropna(subset=[attribute_col])
        print(f"-> Removed {original_na_count} features with missing values.")
        if len(gdf_wgs84) == 0:
             print("!!! Error: No features remaining after removing missing attribute values.")
             return False

    # 11. Get Categories and Save Layers
    categories = sorted(gdf_wgs84[attribute_col].unique())
    print(f"\nFound {len(categories)} unique categories in '{attribute_col}':")
    # print(categories) # Uncomment to list all categories

    print("\nSaving categorized GeoJSON layers...")
    saved_files_count = 0
    skipped_categories = []

    for category in categories:
        print(f"- Processing category: '{category}'")
        category_gdf = gdf_wgs84[gdf_wgs84[attribute_col] == category]

        if category_gdf.empty:
            print("  -> No features for this category, skipping.")
            skipped_categories.append(str(category))
            continue

        # Generate filename
        filename_base = sanitize_filename(category)
        output_filename = f"{filename_base}.geojson"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Save the filtered GeoDataFrame to GeoJSON
            category_gdf.to_file(output_path, driver='GeoJSON', encoding='utf-8')

            # Verify file was created and is not empty
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                 # Optional deeper check: try loading json
                 try:
                      with open(output_path, 'r', encoding='utf-8') as f:
                           geojson_data = json.load(f)
                      if not geojson_data.get('features'):
                           print(f"  -> Warning: Saved file {output_filename} has no 'features' array.")
                           # os.remove(output_path) # Optionally remove invalid file
                           # skipped_categories.append(str(category))
                           # continue
                 except json.JSONDecodeError:
                      print(f"  -> Warning: Saved file {output_filename} is not valid JSON.")
                      # os.remove(output_path) # Optionally remove invalid file
                      # skipped_categories.append(str(category))
                      # continue

                 print(f"  -> Saved {len(category_gdf)} features to {output_filename}")
                 saved_files_count += 1
            else:
                 print(f"  -> Warning: File {output_filename} was not created or is empty.")
                 skipped_categories.append(str(category))

        except Exception as e:
            print(f"  !!! Error saving GeoJSON for category '{category}': {e}")
            skipped_categories.append(str(category))

    print("\n" + "-" * 60)
    print("Processing Summary:")
    print(f"- Total categories found: {len(categories)}")
    print(f"- GeoJSON layers saved successfully: {saved_files_count}")
    if skipped_categories:
        print(f"- Categories skipped (no features or error): {len(skipped_categories)}")
        # print(f"  Skipped: {', '.join(skipped_categories)}") # Uncomment to list skipped
    print(f"- Output directory: {os.path.abspath(output_dir)}")
    print("-" * 60)

    return saved_files_count > 0


# --- Run the process ---
if __name__ == "__main__":
    success = generate_geojson_layers(
        gml_path=GML_FILE_PATH,
        attribute_col=ATTRIBUTE_COLUMN,
        output_dir=OUTPUT_GEOJSON_DIR,
        assumed_crs=ASSUMED_ORIGINAL_CRS,
        simplify=ENABLE_SIMPLIFICATION,
        tolerance=SIMPLIFICATION_TOLERANCE,
        fill_na=FILL_NA_CATEGORY_WITH
    )

    if success:
        print("\nScript finished successfully.")
    else:
        print("\nScript finished with errors or no layers saved.")

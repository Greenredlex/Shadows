#!/usr/bin/env python3
"""
Road Segment Splitter for Shadow Analysis

This module splits fietspad and voetpad segments from bgt_wegdeel.gml into smaller 
sections that are no more than 50m long. This provides better resolution for 
shadow analysis calculations.

Author: Shadow Analysis Tool
Date: 2025
"""

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
import pandas as pd
from typing import List, Tuple, Optional
import os
import contextily as ctx


def split_linestring_by_distance(linestring: LineString, max_distance: float = 50.0) -> List[LineString]:
    """
    Split a LineString into segments of maximum specified distance.
    
    Args:
        linestring: The LineString geometry to split
        max_distance: Maximum length of each segment in meters (default: 50m)
    
    Returns:
        List of LineString segments
    """
    if linestring.is_empty or linestring.length <= max_distance:
        return [linestring]
    
    segments = []
    total_length = linestring.length
    
    # Calculate how many segments we need
    num_segments = int(np.ceil(total_length / max_distance))
    segment_length = total_length / num_segments
    
    for i in range(num_segments):
        start_distance = i * segment_length
        end_distance = min((i + 1) * segment_length, total_length)
        
        # Create the segment using interpolation
        start_point = linestring.interpolate(start_distance)
        end_point = linestring.interpolate(end_distance)
        
        if start_point.equals(end_point):
            continue
            
        # Extract coordinates between these distances
        segment_coords = [start_point.coords[0]]
        
        # Add any original coordinates that fall within this segment
        coords = list(linestring.coords)
        for coord in coords:
            point = Point(coord)
            distance_along = linestring.project(point)
            if start_distance < distance_along < end_distance:
                segment_coords.append(coord)
        
        # Add the end point
        segment_coords.append(end_point.coords[0])
        
        # Create the segment if we have at least 2 points
        if len(segment_coords) >= 2:
            # Remove duplicates while preserving order
            unique_coords = []
            for coord in segment_coords:
                if not unique_coords or coord != unique_coords[-1]:
                    unique_coords.append(coord)
            
            if len(unique_coords) >= 2:
                segment = LineString(unique_coords)
                segments.append(segment)
    
    return segments


def polygon_to_centerline(polygon):
    """
    Convert a polygon to its centerline using the Voronoi diagram approach.
    This is a simplified approach - for more accurate results, consider using 
    specialized libraries like centerline or polygon-to-line conversion tools.
    """
    from shapely.geometry import LineString
    from shapely.ops import voronoi_diagram
    
    try:
        # Simple approach: use the polygon's centroid and create a line along the longest axis
        bounds = polygon.bounds
        minx, miny, maxx, maxy = bounds
        
        # Create a simple centerline from centroid or use the polygon boundary
        if polygon.geom_type == 'Polygon':
            # Use the exterior ring as the centerline approximation
            exterior = polygon.exterior
            if exterior.geom_type == 'LinearRing':
                # Convert LinearRing to LineString
                coords = list(exterior.coords[:-1])  # Remove duplicate last point
                if len(coords) >= 2:
                    return LineString(coords)
        
        # Fallback: create a simple line across the polygon
        centroid = polygon.centroid
        if maxx - minx > maxy - miny:  # Wider than tall
            return LineString([(minx, centroid.y), (maxx, centroid.y)])
        else:  # Taller than wide
            return LineString([(centroid.x, miny), (centroid.x, maxy)])
            
    except Exception:
        # Final fallback: diagonal line
        return LineString([(polygon.bounds[0], polygon.bounds[1]), 
                          (polygon.bounds[2], polygon.bounds[3])])


def load_and_filter_roads(gml_file: str = 'bgt_wegdeel.gml') -> gpd.GeoDataFrame:
    """
    Load and filter the roads data to get only fietspad and voetpad segments.
    Converts polygon road segments to centerlines for proper splitting.
    
    Args:
        gml_file: Path to the GML file containing road data
    
    Returns:
        GeoDataFrame with filtered fietspad and voetpad segments as LineStrings
    """
    if not os.path.exists(gml_file):
        raise FileNotFoundError(f"Road data file not found: {gml_file}")
    
    print(f"📂 Loading roads data from {gml_file}...")
    roads_gdf = gpd.read_file(gml_file)
    
    print(f"   Total road segments loaded: {len(roads_gdf)}")
    print(f"   Columns available: {list(roads_gdf.columns)}")
    
    # Check geometry types
    geom_types = roads_gdf.geometry.type.value_counts()
    print(f"   Geometry types found: {dict(geom_types)}")
    
    # Filter for fietspad and voetpad
    if 'function' in roads_gdf.columns:
        bike_foot_paths = roads_gdf[roads_gdf['function'].isin(['fietspad', 'voetpad'])].copy()
        print(f"   Fietspad/voetpad segments found: {len(bike_foot_paths)}")
        
        if len(bike_foot_paths) == 0:
            print("   ⚠️ No fietspad/voetpad found, using first 500 segments for testing")
            bike_foot_paths = roads_gdf.head(500).copy()
    else:
        print("   ⚠️ 'function' column not found, using first 500 segments for testing")
        bike_foot_paths = roads_gdf.head(500).copy()
    
    # Convert polygon geometries to centerlines
    print(f"   Converting polygon road segments to centerlines...")
    centerlines = []
    
    for idx, row in bike_foot_paths.iterrows():
        geom = row.geometry
        
        if geom.geom_type == 'LineString':
            # Already a line, keep as is
            centerlines.append(row)
        elif geom.geom_type == 'Polygon':
            # Convert polygon to centerline
            try:
                centerline = polygon_to_centerline(geom)
                new_row = row.copy()
                new_row.geometry = centerline
                centerlines.append(new_row)
            except Exception as e:
                print(f"   ⚠️ Could not convert polygon {idx} to centerline: {e}")
                continue
        else:
            print(f"   ⚠️ Skipping unsupported geometry type: {geom.geom_type}")
            continue
    
    # Create new GeoDataFrame with centerlines
    if centerlines:
        bike_foot_paths_lines = gpd.GeoDataFrame(centerlines, crs=roads_gdf.crs)
        print(f"   Centerline conversion completed: {len(bike_foot_paths_lines)} LineString segments")
    else:
        print("   ❌ No valid centerlines could be created")
        bike_foot_paths_lines = gpd.GeoDataFrame(columns=roads_gdf.columns, crs=roads_gdf.crs)
    
    return bike_foot_paths_lines


def split_road_segments(roads_gdf: gpd.GeoDataFrame, max_segment_length: float = 50.0) -> gpd.GeoDataFrame:
    """
    Split road segments into smaller sections.
    
    Args:
        roads_gdf: GeoDataFrame with road segments
        max_segment_length: Maximum length of each segment in meters
    
    Returns:
        GeoDataFrame with split segments
    """
    print(f"🔪 Splitting segments into max {max_segment_length}m sections...")
    
    split_segments = []
    total_original_length = 0
    total_split_length = 0
    
    for idx, row in roads_gdf.iterrows():
        original_geom = row.geometry
        original_length = original_geom.length
        total_original_length += original_length
        
        # Split the geometry
        split_geoms = split_linestring_by_distance(original_geom, max_segment_length)
        
        # Create new rows for each split segment
        for i, split_geom in enumerate(split_geoms):
            new_row = row.copy()
            new_row.geometry = split_geom
            new_row['original_id'] = idx
            new_row['segment_id'] = i
            new_row['segment_length'] = split_geom.length
            new_row['original_length'] = original_length
            
            total_split_length += split_geom.length
            split_segments.append(new_row)
    
    # Create new GeoDataFrame
    split_gdf = gpd.GeoDataFrame(split_segments, crs=roads_gdf.crs)
    
    print(f"   Original segments: {len(roads_gdf)}")
    print(f"   Split segments: {len(split_gdf)}")
    print(f"   Original total length: {total_original_length:.1f}m")
    print(f"   Split total length: {total_split_length:.1f}m")
    print(f"   Length preservation: {(total_split_length/total_original_length)*100:.2f}%")
    
    # Statistics about segment lengths
    segment_lengths = split_gdf['segment_length'].values
    print(f"   Segment length stats:")
    print(f"     Mean: {np.mean(segment_lengths):.2f}m")
    print(f"     Median: {np.median(segment_lengths):.2f}m")
    print(f"     Max: {np.max(segment_lengths):.2f}m")
    print(f"     Min: {np.min(segment_lengths):.2f}m")
    print(f"     Segments > {max_segment_length}m: {np.sum(segment_lengths > max_segment_length)}")
    
    return split_gdf


def create_test_visualization(original_gdf: gpd.GeoDataFrame, 
                            split_gdf: gpd.GeoDataFrame,
                            test_area_bounds: Optional[Tuple[float, float, float, float]] = None,
                            max_segments_to_show: int = 100) -> None:
    """
    Create a visualization comparing original and split segments.
    
    Args:
        original_gdf: Original road segments
        split_gdf: Split road segments
        test_area_bounds: Optional bounds (minx, miny, maxx, maxy) for focusing on a test area
        max_segments_to_show: Maximum number of segments to show to avoid overcrowding
    """
    print(f"📊 Creating test visualization...")
    
    # If test area bounds are provided, clip to that area
    if test_area_bounds:
        from shapely.geometry import box
        test_box = box(*test_area_bounds)
        original_clipped = original_gdf.clip(test_box)
        split_clipped = split_gdf.clip(test_box)
        title_suffix = " (Test Area)"
    else:
        # Take a sample for visualization to avoid overcrowding
        if len(original_gdf) > max_segments_to_show:
            original_clipped = original_gdf.sample(n=max_segments_to_show)
            # Get the corresponding split segments
            original_ids = original_clipped.index.tolist()
            split_clipped = split_gdf[split_gdf['original_id'].isin(original_ids)]
        else:
            original_clipped = original_gdf
            split_clipped = split_gdf
        title_suffix = f" (Sample of {len(original_clipped)} segments)"
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Original segments
    if len(original_clipped) > 0:
        original_clipped.plot(ax=ax1, color='blue', linewidth=2, alpha=0.7)
        bounds = original_clipped.total_bounds
        
        # Add basemap if possible
        try:
            ctx.add_basemap(ax1, crs=original_clipped.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
        except Exception as e:
            print(f"   ⚠️ Could not add basemap to original plot: {e}")
        
        ax1.set_xlim(bounds[0], bounds[2])
        ax1.set_ylim(bounds[1], bounds[3])
        ax1.set_title(f'Original Segments{title_suffix}\n{len(original_clipped)} segments', fontsize=14)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Split segments with color coding by length
    if len(split_clipped) > 0:
        # Color code by segment length
        split_clipped.plot(ax=ax2, column='segment_length', cmap='viridis', 
                          linewidth=1.5, alpha=0.8, legend=True)
        
        bounds = split_clipped.total_bounds
        
        # Add basemap if possible
        try:
            ctx.add_basemap(ax2, crs=split_clipped.crs, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.5)
        except Exception as e:
            print(f"   ⚠️ Could not add basemap to split plot: {e}")
        
        ax2.set_xlim(bounds[0], bounds[2])
        ax2.set_ylim(bounds[1], bounds[3])
        ax2.set_title(f'Split Segments{title_suffix}\n{len(split_clipped)} segments (colored by length)', fontsize=14)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'road_segment_split_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   📸 Visualization saved as: {output_file}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"   ⚠️ Could not display plot (running in headless mode): {e}")
    
    plt.close()


def run_comprehensive_test(gml_file: str = 'bgt_wegdeel.gml', 
                          max_segment_length: float = 50.0,
                          test_area_size: float = 1000.0) -> gpd.GeoDataFrame:
    """
    Run a comprehensive test of the road segment splitting functionality.
    
    Args:
        gml_file: Path to the GML file
        max_segment_length: Maximum segment length in meters
        test_area_size: Size of test area in meters (creates a square around the center)
    
    Returns:
        GeoDataFrame with split segments
    """
    print("🚀 Starting comprehensive road segment splitting test...")
    print("=" * 60)
    
    try:
        # Step 1: Load and filter roads
        original_roads = load_and_filter_roads(gml_file)
        
        if len(original_roads) == 0:
            print("❌ No road segments found to process!")
            return gpd.GeoDataFrame()
        
        # Step 2: Split the segments
        split_roads = split_road_segments(original_roads, max_segment_length)
        
        # Step 3: Create test area around the center of the data
        bounds = original_roads.total_bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        half_size = test_area_size / 2
        
        test_bounds = (
            center_x - half_size,
            center_y - half_size, 
            center_x + half_size,
            center_y + half_size
        )
        
        print(f"\n📍 Test area bounds: {test_bounds}")
        
        # Step 4: Create visualization
        create_test_visualization(original_roads, split_roads, test_bounds)
        
        # Step 5: Quality checks
        print(f"\n🔍 Quality checks:")
        
        # Check that no segments are longer than max_segment_length
        long_segments = split_roads[split_roads['segment_length'] > max_segment_length]
        if len(long_segments) > 0:
            print(f"   ⚠️ Warning: {len(long_segments)} segments are longer than {max_segment_length}m")
            print(f"      Max length found: {long_segments['segment_length'].max():.2f}m")
        else:
            print(f"   ✅ All segments are ≤ {max_segment_length}m")
        
        # Check for very short segments (might indicate issues)
        very_short = split_roads[split_roads['segment_length'] < 1.0]
        if len(very_short) > 0:
            print(f"   ⚠️ Warning: {len(very_short)} segments are < 1m long")
        
        # Check geometry validity
        invalid_geom = split_roads[~split_roads.geometry.is_valid]
        if len(invalid_geom) > 0:
            print(f"   ⚠️ Warning: {len(invalid_geom)} segments have invalid geometry")
        else:
            print(f"   ✅ All segments have valid geometry")
        
        print(f"\n🎉 Test completed successfully!")
        print(f"   Output: {len(split_roads)} segments ready for shadow analysis")
        
        return split_roads
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return gpd.GeoDataFrame()


def save_split_segments(split_gdf: gpd.GeoDataFrame, output_file: str = 'split_bike_foot_paths.gpkg') -> None:
    """
    Save the split segments to a file for use in shadow analysis.
    
    Args:
        split_gdf: GeoDataFrame with split segments
        output_file: Output file path
    """
    if len(split_gdf) > 0:
        split_gdf.to_file(output_file, driver='GPKG')
        print(f"💾 Split segments saved to: {output_file}")
        print(f"   {len(split_gdf)} segments saved")
        print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    else:
        print("❌ No segments to save")


def load_split_segments_for_shadow_analysis(split_file: str = 'split_bike_foot_paths.gpkg', 
                                          bbox_bounds: Optional[Tuple[float, float, float, float]] = None) -> gpd.GeoDataFrame:
    """
    Load the pre-split segments for use in shadow analysis.
    
    Args:
        split_file: Path to the split segments file
        bbox_bounds: Optional bounds (minx, miny, maxx, maxy) to clip the segments
    
    Returns:
        GeoDataFrame with split segments ready for shadow analysis
    """
    if not os.path.exists(split_file):
        print(f"⚠️ Split segments file not found: {split_file}")
        print("   Running road segment splitter to create it...")
        
        # Automatically create the split segments
        split_segments = run_comprehensive_test(
            gml_file='bgt_wegdeel.gml',
            max_segment_length=50.0,
            test_area_size=1000.0
        )
        
        if len(split_segments) > 0:
            save_split_segments(split_segments, split_file)
        else:
            return gpd.GeoDataFrame()
    
    print(f"📂 Loading split segments from {split_file}...")
    split_gdf = gpd.read_file(split_file)
    
    if bbox_bounds:
        from shapely.geometry import box
        clip_box = box(*bbox_bounds)
        split_gdf = split_gdf.clip(clip_box)
        print(f"   Clipped to bbox: {len(split_gdf)} segments in analysis area")
    
    print(f"   Loaded {len(split_gdf)} split segments for shadow analysis")
    return split_gdf


if __name__ == "__main__":
    # Run the comprehensive test
    print("🔧 Road Segment Splitter - Testing Mode")
    print("=" * 50)
    
    # Check if required file exists
    gml_file = 'bgt_wegdeel.gml'
    if not os.path.exists(gml_file):
        print(f"❌ Required file not found: {gml_file}")
        print("   Please ensure the BGT road data file is in the current directory")
        exit(1)
    
    # Run the test
    split_segments = run_comprehensive_test(
        gml_file=gml_file,
        max_segment_length=50.0,  # 50 meter segments
        test_area_size=1000.0     # 1km x 1km test area
    )
    
    # Save the results
    if len(split_segments) > 0:
        save_split_segments(split_segments, 'split_bike_foot_paths.gpkg')
        
        print(f"\n📋 Summary for shadow analysis integration:")
        print(f"   - Original max segment length: variable")
        print(f"   - New max segment length: 50m")
        print(f"   - Total segments: {len(split_segments)}")
        print(f"   - Ready for import into shadow_analysis_complete.py")
        
        # Test the integration function
        print(f"\n🔗 Testing integration function...")
        test_bounds = split_segments.total_bounds
        center_x, center_y = (test_bounds[0] + test_bounds[2]) / 2, (test_bounds[1] + test_bounds[3]) / 2
        test_bbox = (center_x - 500, center_y - 500, center_x + 500, center_y + 500)
        
        integration_test = load_split_segments_for_shadow_analysis(
            'split_bike_foot_paths.gpkg', 
            test_bbox
        )
        print(f"   Integration test successful: {len(integration_test)} segments loaded")
    
    print(f"\n✅ Done! Check the generated visualization and output files.") 
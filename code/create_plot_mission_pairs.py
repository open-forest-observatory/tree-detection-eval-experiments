"""
This script downloads mission metadata files from S3 using rclone.
Before running the script, set RCLONE_S3_ACCESS_KEY_ID and RCLONE_S3_SECRET_ACCESS_KEY env variables with appropriate credentials.
Then run the script to get a list of mission-plot pairs based on spatial overlap and closest survey/mission dates. 
The output is saved to a CSV file that can be used by ofo-argo/argo-workflows/tree-detection-and-eval.yaml as an input file. 
"""

import shutil
import subprocess
import pyproj
import pandas as pd
import geopandas as gpd
from pathlib import Path

def ensure_projected_crs(gdf):
    if gdf.crs.is_projected:
        return gdf
    if gdf.crs != pyproj.CRS.from_epsg(4326):
        gdf = gdf.to_crs(epsg=4326)
    centroid = gdf.geometry.iloc[0].centroid
    lon, lat = centroid.x, centroid.y
    if lon > 0:
        lon = -lon
    epsg = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    return gdf.to_crs(epsg=epsg)

def parse_survey_date(val):
    """Helper function to parse survey date from the plot metadata which is in inconsistent formats."""
    s = str(int(val)) if pd.notna(val) else ""
    if len(s) == 4:
        return pd.to_datetime(s + "-07-01")
    elif len(s) == 6:
        return pd.to_datetime(s, format="%Y%m")
    elif len(s) == 8:
        return pd.to_datetime(s, format="%Y%m%d")
    return pd.NaT

# Minimum plot area to be considered for inclsuion in evaluation (square meters)
AREA_THRESHOLD_M2 = 500
# Buffer distance added to the plot bounds before finding overlapping missions (meters)
PLOT_BUFFER_M = 20

# Remote folder containing mission metadata files
ALL_MISSIONS_REMOTE_FOLDER = "js2s3:ofo-public/drone/missions_03"
# Output file path containing mission-plot pairs
OUTPUT_FILE = "/ofo-share/argo-data/argo-input/tree-detection-and-evaluation/datasets.csv"
# File containing list of plot IDs to be potentially included in evaluation
PLOT_IDS_FILE = "/ofo-share/project-data/species-prediction-project/raw/withheld_ground_plot_ids_v1.csv"
# File containing ground plot boundaries and survey dates
GROUND_REFERENCE_PLOTS_FILE = "/ofo-share/argo-data/argo-input/tree-detection-and-evaluation/ofo_ground-reference_plots.gpkg"
# This is where rclone downloads mission metadata files from s3 for matching with plots. It gets deleted at the end of the script.
MISSION_METADATA_FOLDERS = "/ofo-share/argo-data/argo-input/tree-detection-and-evaluation/mission-metadata"

# Load the plot IDs
all_plot_ids = pd.read_csv(
    PLOT_IDS_FILE,
    dtype=str
)["plot_id"].tolist()

# Load and filter plot bounds
plot_bounds = gpd.read_file(GROUND_REFERENCE_PLOTS_FILE)
plot_bounds["plot_id"] = plot_bounds["plot_id"].astype(str)
plot_bounds = plot_bounds[plot_bounds["plot_id"].isin(all_plot_ids)]

# Download mission metadata from S3
Path(MISSION_METADATA_FOLDERS).mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "rclone", "copy",
        ALL_MISSIONS_REMOTE_FOLDER,
        MISSION_METADATA_FOLDERS,
        "--include", "*_mission-metadata.gpkg",
    ],
    check=True,
)

# Load mission metadata files and concatenate into a single GeoDataFrame
mission_files = list(Path(MISSION_METADATA_FOLDERS).rglob("*.gpkg"))
mission_metadata = gpd.GeoDataFrame(
    pd.concat([gpd.read_file(f) for f in mission_files]),
    crs=gpd.read_file(mission_files[0]).crs
)

# Project both to metric CRS
mission_metadata = ensure_projected_crs(mission_metadata)
plot_bounds = plot_bounds.to_crs(mission_metadata.crs)

# Filter out plots below area threshold
plot_bounds = plot_bounds[plot_bounds.geometry.area >= AREA_THRESHOLD_M2]
print(f"{len(plot_bounds)} plots remaining after filtering by area >= {AREA_THRESHOLD_M2}")

# Buffer by 20m and do spatial join
plot_bounds_buffered = plot_bounds.copy()
plot_bounds_buffered["geometry"] = plot_bounds_buffered.geometry.buffer(PLOT_BUFFER_M)

# "within" is used to ensure that the plot bounds are fully within the mission bounds.
pairs = gpd.sjoin(
    plot_bounds_buffered[["plot_id", "geometry"]],
    mission_metadata[["mission_id", "geometry"]],
    how="inner",
    predicate="within"
)[["plot_id", "mission_id"]]

# Parse plot survey dates
plot_survey_dates = plot_bounds[["plot_id", "survey_date"]].copy()
plot_survey_dates["survey_date_parsed"] = plot_survey_dates["survey_date"].apply(parse_survey_date)

# Parse mission dates
mission_dates = mission_metadata[["mission_id", "earliest_date_derived"]].copy()
mission_dates["earliest_date_derived"] = pd.to_datetime(mission_dates["earliest_date_derived"], errors="coerce")

# Merge dates onto pairs and compute difference
pairs = pairs.merge(plot_survey_dates[["plot_id", "survey_date_parsed"]], on="plot_id", how="left")
pairs = pairs.merge(mission_dates, on="mission_id", how="left")
pairs["date_diff_days"] = (pairs["earliest_date_derived"] - pairs["survey_date_parsed"]).abs().dt.days

# Keep only the closest mission per plot
pairs = (
    pairs.sort_values("date_diff_days")
         .drop_duplicates(subset="plot_id", keep="first")
         .reset_index(drop=True)
)

# Save the pairs to a CSV file
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
pairs[["mission_id", "plot_id"]].to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(pairs)} pairs to {OUTPUT_FILE}")

# Clean up downloaded mission metadata files
shutil.rmtree(MISSION_METADATA_FOLDERS)
print(f"Deleted {MISSION_METADATA_FOLDERS}")

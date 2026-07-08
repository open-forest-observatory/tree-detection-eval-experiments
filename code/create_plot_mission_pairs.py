"""
This script downloads mission metadata files from S3 using rclone.
Before running the script, set RCLONE_S3_ACCESS_KEY_ID and RCLONE_S3_SECRET_ACCESS_KEY env variables with appropriate credentials.
Then run the script to get a list of mission-plot pairs based on spatial overlap and closest survey/mission dates.
Missions are classified as high-nadir/low-oblique, and plots are only paired with
high-nadir missions. Plots with no overlapping high-nadir mission are dropped and reported.
The output is saved to a CSV file that can be used by ofo-argo/argo-workflows/tree-detection-and-eval.yaml as an input file.
"""

import shutil
import subprocess
import pyproj
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

def classify_mission(row):
    """Classify a mission as high-nadir, low-oblique, or other, mirroring
    tree-species-prediction/1_data_prep/04_pair_drone_with_ground.py::classify_mission."""
    altitude = row["mean_altitude"]
    pitch = row["camera_pitch_derived"]
    terrain_corr = row["flight_terrain_correlation_photogrammetry"]
    sd_photogrammetry_altitude = row["sd_photogrammetry_altitude"]
    front_overlap = row["overlap_front_nominal"]
    side_overlap = row["overlap_side_nominal"]

    # Check for NaNs
    if (
        pd.isna(altitude)
        or pd.isna(pitch)
        or pd.isna(terrain_corr)
        or pd.isna(front_overlap)
        or pd.isna(side_overlap)
    ):
        return "unknown"

    # Must meet terrain fidelity or SD requirement
    if not (terrain_corr > 0.75 or sd_photogrammetry_altitude < 12):
        return "low-terrain-fidelity"

    # High-Nadir requirements
    if (
        100 <= altitude <= 160
        and 0 <= pitch <= 10
        and (
            (front_overlap >= 90 and side_overlap >= 80)
            or (front_overlap >= 85 and side_overlap >= 85)
        )
    ):
        return "high-nadir"

    # Low-Oblique requirements
    if (
        60 <= altitude <= 120
        and 18 <= pitch <= 38
        and front_overlap >= 70
        and side_overlap >= 60
    ):
        return "low-oblique"

    return "unclassified"

def extract_min_overlap(val):
    """Helper function to get minimum value from comma-separated overlap values."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, str) and "," in val:
        return min(float(x.strip()) for x in val.split(","))
    return float(val)

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
# File containing all missions metadata (with derived altitude fields) used to classify missions as nadir/oblique
MISSION_CLASSIFICATION_FILE = "/ofo-share/project-data/species-prediction-project/intermediate/preprocessing/ofo-all-missions-metadata-with-altitude.gpkg"
# File containing manual quality assessments of the drone-to-field registration shift for each mission-plot pair
SHIFT_EVAL_FILE = "/ofo-share/repos/amritha/tree-detection-eval-experiments/data/drone-field-shift-eval.csv"
# Shift eval quality columns
SHIFT_EVAL_QUALITY_COLS = [
    "quality_of_predicted_registration",
    "confidence_in_assessment",
    "quality_of_field_trees",
]
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

# Load mission metadata with derived altitude fields and classify each mission as
# high-nadir / low-oblique / other, so plots can be paired with nadir missions only
mission_classification = gpd.read_file(MISSION_CLASSIFICATION_FILE)
mission_classification["mission_id"] = mission_classification["mission_id"].astype(str)
mission_classification["overlap_front_nominal"] = mission_classification["overlap_front_nominal"].apply(extract_min_overlap)
mission_classification["overlap_side_nominal"] = mission_classification["overlap_side_nominal"].apply(extract_min_overlap)
mission_classification["camera_pitch_derived"] = pd.to_numeric(mission_classification["camera_pitch_derived"], errors="coerce")
mission_classification["flight_terrain_correlation_photogrammetry"] = pd.to_numeric(mission_classification["flight_terrain_correlation_photogrammetry"], errors="coerce")
mission_classification["mission_type"] = mission_classification.apply(classify_mission, axis=1)

# Keep only missions classified as high-nadir
mission_metadata["mission_id"] = mission_metadata["mission_id"].astype(str)
mission_metadata = mission_metadata.merge(
    mission_classification[["mission_id", "mission_type"]], on="mission_id", how="left"
)
mission_metadata = mission_metadata[mission_metadata["mission_type"] == "high-nadir"]
print(f"{len(mission_metadata)} missions remaining after filtering to high-nadir only")

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

# Report and drop plots that have no overlapping high-nadir mission to pair with
paired_plot_ids = set(pairs["plot_id"])
unpaired_plot_ids = sorted(set(plot_bounds["plot_id"]) - paired_plot_ids)
if unpaired_plot_ids:
    print(f"{len(unpaired_plot_ids)} plot(s) dropped due to no overlapping high-nadir mission:")
    for plot_id in unpaired_plot_ids:
        print(f"  {plot_id}")

# Load shift eval quality assessments and align ID formats with the pairs (zero-padded strings)
shift_eval = pd.read_csv(SHIFT_EVAL_FILE, dtype=str)
shift_eval["mission_id"] = shift_eval["drone_mission"].str.zfill(6)
shift_eval["plot_id"] = shift_eval["plot_id"].str.zfill(4)
# Convert the quality columns from str to numbers
for col in SHIFT_EVAL_QUALITY_COLS:
    shift_eval[col] = pd.to_numeric(shift_eval[col], errors="coerce")

# Match each mission-plot pair to its shift eval row
pairs = pairs.merge(
    shift_eval[["mission_id", "plot_id"] + SHIFT_EVAL_QUALITY_COLS],
    on=["mission_id", "plot_id"],
    how="left",
    indicator=True,
)
# Report and drop pairs with no matching shift eval entry
missing_shift_eval = pairs[pairs["_merge"] == "left_only"]
if len(missing_shift_eval):
    print(f"{len(missing_shift_eval)} mission-plot pair(s) not found in shift eval file, dropping:")
    for _, row in missing_shift_eval.iterrows():
        print(f"mission_id={row['mission_id']}, plot_id={row['plot_id']}")
pairs = pairs[pairs["_merge"] == "both"].drop(columns="_merge")

# Report and drop pairs with poor shift quality (any quality column rated 1 or 2)
poor_shift_quality = pairs[SHIFT_EVAL_QUALITY_COLS].isin([1, 2]).any(axis=1)
if poor_shift_quality.any():
    print(f"{poor_shift_quality.sum()} mission-plot pair(s) dropped due to poor shift quality:")
    for _, row in pairs[poor_shift_quality].iterrows():
        print(f"mission_id={row['mission_id']}, plot_id={row['plot_id']}")
pairs = pairs[~poor_shift_quality].drop(columns=SHIFT_EVAL_QUALITY_COLS).reset_index(drop=True)

# Save the pairs to a CSV file
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
pairs[["mission_id", "plot_id"]].to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(pairs)} pairs to {OUTPUT_FILE}")

# Clean up downloaded mission metadata files
shutil.rmtree(MISSION_METADATA_FOLDERS)
print(f"Deleted {MISSION_METADATA_FOLDERS}")

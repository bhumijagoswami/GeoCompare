# =============================================================================
# GeoCompare — Train a NEW LULC classifier on Google Colab
#
# HOW TO USE:
# 1. Open a new notebook at https://colab.research.google.com
# 2. Paste each "# --- CELL n ---" block into its own cell, in order, and run.
# 3. At the end, download model.pkl and model_metadata.json and replace the
#    versions in your GeoCompare repo.
#
# WHAT THIS DOES:
# Instead of relying on an unknown pre-trained model.pkl, this script builds
# a fresh training set by using ESA WorldCover (a well-validated global LULC
# product) as ground truth, sampled across multiple diverse regions, then
# trains a RandomForestClassifier on the same Sentinel-2 + index features
# GeoCompare already extracts at inference time (gee_utils.py).
#
# This gives you: known label provenance, a reproducible train/test split,
# and real accuracy numbers to put in model_metadata.json.
# =============================================================================

# --- CELL 1: Install deps (Colab already has most of these, but pin ee) ---
!pip install -q earthengine-api scikit-learn pandas numpy joblib

# --- CELL 2: Authenticate Earth Engine ---
import ee

PROJECT_ID = "earth-engine-demo-501808"  # <-- change to your own GEE project ID

ee.Authenticate()  # opens an interactive auth flow in Colab
ee.Initialize(project=PROJECT_ID)

# --- CELL 3: Config — feature bands, classes, and sample regions ---
FEATURE_COLS = ['B2', 'B3', 'B4', 'B8', 'B11', 'NDVI', 'NDWI', 'NDBI']
CLASS_NAMES = {0: "Vegetation", 1: "Built-up", 2: "Water", 3: "Bare Land"}

# WorldCover codes -> GeoCompare class ids (must match map_utils.py's remap
# exactly, so the map overlay and the model agree on what each class means):
#   10 Tree cover, 40 Cropland      -> 0 Vegetation
#   50 Built-up                     -> 1 Built-up
#   80 Permanent water bodies       -> 2 Water
#   60 Bare / sparse vegetation     -> 3 Bare Land
WORLDCOVER_SRC = [10, 40, 50, 80, 60]
WORLDCOVER_DST = [0, 0, 1, 2, 3]

# Diverse sample regions (lat, lon, radius_m) — mix of urban, agricultural,
# arid, and water-adjacent areas so the model isn't overfit to one geography.
# Add/remove regions freely; more regions = better generalization.
SAMPLE_REGIONS = [
    {"name": "Delhi_NCR_urban",        "lat": 28.6139, "lon": 77.2090, "radius": 20000},
    {"name": "Jaipur_urban_arid",      "lat": 26.9124, "lon": 75.7873, "radius": 20000},
    {"name": "Punjab_cropland",        "lat": 30.7333, "lon": 76.7794, "radius": 20000},
    {"name": "Thar_desert_bare",       "lat": 27.0238, "lon": 71.0025, "radius": 20000},
    {"name": "Chilika_lake_water",     "lat": 19.7000, "lon": 85.3200, "radius": 15000},
    {"name": "Western_Ghats_veg",      "lat": 15.3173, "lon": 74.1240, "radius": 20000},
    {"name": "Mumbai_dense_urban",     "lat": 19.0760, "lon": 72.8777, "radius": 15000},
    {"name": "Ganga_floodplain_mixed", "lat": 25.3176, "lon": 82.9739, "radius": 20000},
]

START_DATE, END_DATE = "2023-01-01", "2023-12-31"
CLOUD_FILTER = 10
POINTS_PER_CLASS_PER_REGION = 150  # stratified sample size

# --- CELL 4: Build the same feature image GeoCompare uses at inference time ---
def get_feature_image(lat, lon, radius_m):
    poi = ee.Geometry.Point([lon, lat]).buffer(radius_m)
    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
    )
    if dataset.size().getInfo() == 0:
        return None, None
    image = dataset.median().clip(poi)
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11'])
    feature_img = bands.addBands([ndvi, ndwi, ndbi])

    worldcover = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(poi)
    label_img = worldcover.remap(WORLDCOVER_SRC, WORLDCOVER_DST, defaultValue=255).rename('label')

    return feature_img.addBands(label_img), poi

# --- CELL 5: Stratified sampling per region (label-aware, via stratifiedSample) ---
import pandas as pd
import time

all_rows = []

for region in SAMPLE_REGIONS:
    print(f"Sampling {region['name']}...")
    combined_img, poi = get_feature_image(region["lat"], region["lon"], region["radius"])
    if combined_img is None:
        print(f"  Skipped (no imagery available).")
        continue

    try:
        samples = combined_img.stratifiedSample(
            numPoints=POINTS_PER_CLASS_PER_REGION,
            classBand='label',
            region=poi,
            scale=30,
            classValues=[0, 1, 2, 3, 255],
            classPoints=[POINTS_PER_CLASS_PER_REGION] * 4 + [0],  # skip label 255 (unmapped)
            geometries=False,
            seed=42,
        )
        data = samples.getInfo()
        rows = [f['properties'] for f in data['features']]
        for r in rows:
            r['region'] = region['name']
        all_rows.extend(rows)
        print(f"  Got {len(rows)} samples.")
    except Exception as e:
        print(f"  Failed: {e}")
    time.sleep(1)  # be polite to the API between regions

df = pd.DataFrame(all_rows)
print(f"\nTotal samples collected: {len(df)}")
print(df['label'].value_counts().rename(index=CLASS_NAMES))

# --- CELL 6: Clean + inspect before training ---
df = df.dropna(subset=FEATURE_COLS + ['label'])
df = df[df['label'] != 255]
print(f"Samples after cleaning: {len(df)}")
print(df.groupby('label')[FEATURE_COLS].mean().rename(index=CLASS_NAMES))

# --- CELL 7: Train/test split + train the model ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = df[FEATURE_COLS]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)])

print(f"Test accuracy: {accuracy:.4f}\n")
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("\nClassification report:")
print(report)

# --- CELL 8: Export model.pkl + a filled-in model_metadata.json ---
import joblib
import json
from datetime import date

joblib.dump(model, "model.pkl")

class_dist = df['label'].value_counts(normalize=True).rename(index=CLASS_NAMES).round(4).to_dict()

metadata = {
    "model_file": "model.pkl",
    "model_type": "RandomForestClassifier",
    "sklearn_version": __import__("sklearn").__version__,
    "trained_on": date.today().isoformat(),
    "training_script": "train_model_colab.py",
    "training_data_source": (
        "ESA WorldCover v200 2021 used as ground-truth labels, remapped to 4 classes; "
        "Sentinel-2 SR Harmonized (2023 median composite) used for feature bands. "
        f"Sampled across {len(SAMPLE_REGIONS)} regions: "
        f"{', '.join(r['name'] for r in SAMPLE_REGIONS)}."
    ),
    "feature_columns": FEATURE_COLS,
    "class_mapping": {str(k): v for k, v in CLASS_NAMES.items()},
    "class_distribution_in_training_set": {k: f"{v*100:.1f}%" for k, v in class_dist.items()},
    "validation_accuracy": round(float(accuracy), 4),
    "confusion_matrix": cm.tolist(),
    "n_train_samples": len(X_train),
    "n_test_samples": len(X_test),
    "notes": "Trained in Colab via train_model_colab.py using WorldCover-derived labels for traceability."
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved model.pkl and model_metadata.json")

# --- CELL 9: Download the files to your machine ---
from google.colab import files
files.download("model.pkl")
files.download("model_metadata.json")

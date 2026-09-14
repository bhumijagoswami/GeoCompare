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
!pip install -q earthengine-api scikit-learn pandas numpy joblib xgboost

# --- CELL 2: Authenticate Earth Engine ---
import ee

PROJECT_ID = "earth-engine-demo-501808"  # <-- change to your own GEE project ID

ee.Authenticate()  # opens an interactive auth flow in Colab
ee.Initialize(project=PROJECT_ID)

# --- CELL 3: Config — feature bands, classes, and sample regions ---
# v2: added B12, BSI, and TEXTURE to reduce Built-up <-> Bare Land confusion.
# v5: added Seasonal NDVI, Sentinel-1 SAR (VV/VH), and VIIRS Nighttime Lights.
FEATURE_COLS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI', 'BSI', 'TEXTURE', 'NDVI_STD', 'SAR_VV', 'SAR_VH', 'NIGHT_LIGHTS']
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
# v3: expanded from 8 to 16 regions specifically to push accuracy toward 80%+
# by giving the model more diverse, higher-volume training data.
SAMPLE_REGIONS = [
    {"name": "Delhi_NCR_urban",        "lat": 28.6139, "lon": 77.2090, "radius": 20000},
    {"name": "Jaipur_urban_arid",      "lat": 26.9124, "lon": 75.7873, "radius": 20000},
    {"name": "Punjab_cropland",        "lat": 30.7333, "lon": 76.7794, "radius": 20000},
    {"name": "Thar_desert_bare",       "lat": 27.0238, "lon": 71.0025, "radius": 20000},
    {"name": "Chilika_lake_water",     "lat": 19.7000, "lon": 85.3200, "radius": 15000},
    {"name": "Western_Ghats_veg",      "lat": 15.3173, "lon": 74.1240, "radius": 20000},
    {"name": "Mumbai_dense_urban",     "lat": 19.0760, "lon": 72.8777, "radius": 15000},
    {"name": "Ganga_floodplain_mixed", "lat": 25.3176, "lon": 82.9739, "radius": 20000},
    {"name": "Bangalore_urban",        "lat": 12.9716, "lon": 77.5946, "radius": 18000},
    {"name": "Chennai_coastal_urban",  "lat": 13.0827, "lon": 80.2707, "radius": 18000},
    {"name": "Sundarbans_water_mangrove", "lat": 21.9497, "lon": 88.9468, "radius": 15000},
    {"name": "Rann_of_Kutch_bare_salt", "lat": 23.8859, "lon": 69.8593, "radius": 20000},
    {"name": "Himalayan_foothill_veg", "lat": 30.3165, "lon": 78.0322, "radius": 20000},
    {"name": "Godavari_delta_cropland", "lat": 16.9891, "lon": 81.7800, "radius": 20000},
    {"name": "Rajasthan_bare_arid",    "lat": 25.2138, "lon": 73.0299, "radius": 20000},
    {"name": "Kerala_backwaters_water", "lat": 9.4981, "lon": 76.3388, "radius": 15000},
]

START_DATE, END_DATE = "2023-01-01", "2023-12-31"
CLOUD_FILTER = 10
POINTS_PER_CLASS_PER_REGION = 500

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

    def calc_ndvi(img):
        return img.normalizedDifference(['B8', 'B4']).rename('NDVI_STD')
    ndvi_std = dataset.map(calc_ndvi).reduce(ee.Reducer.stdDev()).rename('NDVI_STD')

    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(poi)
          .filterDate(START_DATE, END_DATE)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .median()
          .select(['VV', 'VH'], ['SAR_VV', 'SAR_VH']))

    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterBounds(poi)
             .filterDate(START_DATE, END_DATE)
             .median()
             .select(['avg_rad'], ['NIGHT_LIGHTS']))

    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

    bsi = image.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {
            'SWIR1': image.select('B11'),
            'RED': image.select('B4'),
            'NIR': image.select('B8'),
            'BLUE': image.select('B2'),
        }
    ).rename('BSI')

    texture = image.select('B8').reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(radius=1),
    ).rename('TEXTURE')

    bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    
    feature_img = bands.addBands([ndvi, ndwi, ndbi, bsi, texture, ndvi_std, s1, viirs])

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
            classPoints=[POINTS_PER_CLASS_PER_REGION] * 4 + [0], 
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
    time.sleep(1)

df = pd.DataFrame(all_rows)
print(f"\nTotal samples collected: {len(df)}")
print(df['label'].value_counts().rename(index=CLASS_NAMES))

# --- CELL 6: Clean + inspect before training ---
df = df.dropna(subset=FEATURE_COLS + ['label'])
df = df[df['label'] != 255]
print(f"Samples after cleaning: {len(df)}")
print(df.groupby('label')[FEATURE_COLS].mean().rename(index=CLASS_NAMES))

# --- CELL 7: Train/test split + XGBoost ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

X = df[FEATURE_COLS]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

base_model = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    tree_method='hist',
)

print("Running grid search for XGBoost (this can take a few minutes)...")
grid_search = GridSearchCV(
    base_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train, sample_weight=sample_weights)

model = grid_search.best_estimator_
print(f"\nBest params: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)])

print(f"\nHeld-out test accuracy: {accuracy:.4f}\n")
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
    "model_type": "XGBClassifier",
    "sklearn_version": __import__("sklearn").__version__,
    "xgboost_version": __import__("xgboost").__version__,
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
    "cross_validation_accuracy": round(float(grid_search.best_score_), 4),
    "best_hyperparameters": grid_search.best_params_,
    "confusion_matrix": cm.tolist(),
    "n_train_samples": len(X_train),
    "n_test_samples": len(X_test),
    "notes": (
        "Trained in Colab via train_model_colab.py using WorldCover-derived labels for traceability. "
        "v1 (RF, 8 features, 8 regions): 73.85% accuracy. "
        "v2 (RF, +B12/BSI/TEXTURE, 8 regions): 76.88%. "
        "v3 (RF, 16 regions, GridSearchCV tuning, class_weight='balanced'): 77.92%. "
        "v4: XGBClassifier baseline. "
        "v5 (this version): Added NDVI_STD, SAR_VV, SAR_VH, and NIGHT_LIGHTS to resolve Built-up/Bare Land confusion."
    )
}

with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Saved model.pkl and model_metadata.json")

# --- CELL 9: Download the files to your machine ---
from google.colab import files
files.download("model.pkl")
files.download("model_metadata.json")

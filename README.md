

```markdown
# 🌍 GeoCompare

**v3: Single-location concentric ring analysis, across multiple years, combining land-cover change (LULC) with air quality (NO2).**

GeoCompare shows how one location has changed over time — not just spatially (center → outskirts, via concentric rings) but temporally (year over year), and now correlates that change with **air quality trends**.

## How it works

1. Pick one location (lat/lon) and a max radius → it is split into **3 true concentric annulus rings** (not solid nested circles — each ring's stats reflect only that zone).
2. Pick 2+ years (2019 onward, since that's when Sentinel-5P NO2 data starts).
3. For each ring, for each year, it computes:
   - **LULC composition** (Vegetation / Built-up / Water / Bare Land %) via an XGBoost classifier trained on Sentinel-2 optical bands, Sentinel-1 SAR (VV/VH), VIIRS nighttime lights, and ESA WorldCover labels (see `train_model_colab.py`).
   - **Mean NO2 concentration** (µmol/m²) via Sentinel-5P TROPOMI — a raw satellite measurement, no ML model involved.
4. Renders per-year maps, per-ring LULC trend charts, an NO2 trend chart, and a headline "change from first year to last year" table per ring — the basis for spotting patterns like "this ring's Built-up % rose sharply, and its NO2 rose alongside it."

## Data sources & real limitations (know these before demoing)

- **Sentinel-2, Sentinel-1 SAR, & VIIRS** (LULC features): available across multi-year comparisons to construct a 15-band feature stack.
- **Sentinel-5P TROPOMI** (NO2): available from **2018 onward only** — this is why the app does not allow years before 2019 (2018 itself is a partial year of coverage). There is **no way to get satellite-based NO2 for earlier years (e.g., 2011)** — that data simply does not exist.
- **ESA WorldCover** (used for the map's visual LULC reference layer) only has **2020 and 2021** editions. It is a *static* reference layer on the map — it will look the same regardless of which year you select. The actual year-by-year LULC percentages come from our own trained model applied to each year's satellite composite, not from this overlay.
- The model was trained across **16 diverse eco-regions in India** (arid, urban, coastal, delta, and foothill zones) — treat its accuracy as most trustworthy within similar Indian landscapes, not globally validated.

## Project structure

```"
geocompare/
├── app.py                  # Streamlit UI — single location, multi-year
├── gee_utils.py            # Earth Engine auth, feature composite extraction, ring geometry
├── analysis.py             # XGBoost classification, true-annulus rings, year-loop analysis
├── map_utils.py            # Folium map rendering, legend, ring overlays
├── train_model_colab.py    # Colab script: stratified sampling & XGBoost training pipeline
├── model.pkl               # Trained XGBClassifier (binary)
├── model_metadata.json     # Training provenance, accuracy, confusion matrix
├── requirements.txt
├── .gitignore
└── README.md

```

## Model performance

Current model: **83.27% accuracy** (15 features: 6 spectral bands + NDVI/NDWI/NDBI + BSI + local texture + seasonal NDVI std + SAR VV/VH + VIIRS nighttime lights).

* **Validation Accuracy:** 83.27%
* **Cross-Validation Accuracy:** 83.55%
* **Per-class recall:** Water 94.1%, Vegetation 87.2%, Built-up 75.3%, Bare Land 74.7%.

**Progression:**

* **v1 (RF, 8 features, 8 regions):** 73.85% accuracy.
* **v2 (RF, +B12/BSI/TEXTURE, 8 regions):** 76.88% accuracy.
* **v3 (RF, 16 regions, GridSearch):** 77.92% — doubling data revealed an algorithm ceiling.
* **v4 (XGBoost baseline, 11 features):** 78.54% — slight improvement, but Built-up and Bare Land remained heavily confused.
* **v5 (Current — XGBoost + SAR + VIIRS + Seasonal NDVI, 16 regions):** 83.27% — SAR double-bounce radar scattering and nighttime radiance successfully separated concrete from dry soil.

See `model_metadata.json` for the full confusion matrix, hyperparameter specs, and split distributions.

## Setup

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt

```


2. Set up Earth Engine auth. For local dev, run `earthengine authenticate` once.
For Streamlit Cloud, add a service account under **Settings → Secrets**:
```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "..."
client_email = "..."
client_id = "..."

```


The service account needs these IAM roles on your Google Cloud project:
* **Service Usage Consumer**
* **Earth Engine Resource Viewer**
* **Earth Engine Resource Writer** (required for map tile generation via `getMapId`, not just data reads).


3. Ensure `model.pkl` is present in the repo root and `model_metadata.json` reflects the current training run.
4. Run locally:
```bash
streamlit run app.py

```



## Known limitations / roadmap

* **Sensor Resolution Discrepancy**: Sentinel-5P NO2 (~1.1 km resolution) and Sentinel-2/Sentinel-1 (~10–30 m resolution) operate at fundamentally different spatial scales. Ring-level aggregation smooths this, but pixels are not 1:1 comparable.
* **Descriptive Correlation**: Trends showing simultaneous increases in Built-up % and NO2 represent observed spatial-temporal correlations, not a mathematically proven causal model.
* **Static Map Overlay**: Live inference tiles cannot be streamed pixel-by-pixel into Folium without server-side Earth Engine asset ingestion; Folium displays the static ESA WorldCover 2021 layer for visual reference while the graphs display the actual live model predictions.

## Contributing / editing on GitHub

* Never commit `.streamlit/secrets.toml` or any private service-account key files.
* If you retrain `model.pkl`, update `model_metadata.json` and this README's "Model performance" section in the same commit.
* If you change `FEATURE_COLS` in `analysis.py`, you **must** update `train_model_colab.py` and `gee_utils.py` to match identically, or prediction pipelines will fail with a dimension mismatch.

```

```

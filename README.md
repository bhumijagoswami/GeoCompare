# 🌍 GeoCompare

**v3: Single-location concentric ring analysis, across multiple years, combining land-cover change (LULC) with air quality (NO2).**

GeoCompare shows how one location has changed over time — not just spatially
(center → outskirts, via concentric rings) but temporally (year over year),
and now correlates that change with **air quality trends**.

## How it works

1. Pick one location (lat/lon) and a max radius → it's split into **3 true
   concentric annulus rings** (not solid nested circles — each ring's stats
   reflect only that zone).
2. Pick 2+ years (2019 onward, since that's when Sentinel-5P NO2 data starts).
3. For each ring, for each year, it computes:
   - **LULC composition** (Vegetation / Built-up / Water / Bare Land %) via
     a Random Forest classifier trained on Sentinel-2 imagery + ESA WorldCover
     labels (see `train_model_colab.py`).
   - **Mean NO2 concentration** (µmol/m²) via Sentinel-5P TROPOMI — a raw
     satellite measurement, no ML model involved.
4. Renders per-year maps, per-ring LULC trend charts, an NO2 trend chart, and
   a headline "change from first year to last year" table per ring — the
   basis for spotting patterns like "this ring's Built-up % rose sharply,
   and its NO2 rose alongside it."

## Data sources & real limitations (know these before demoing)

- **Sentinel-2** (LULC features): available from mid-2015 onward.
- **Sentinel-5P TROPOMI** (NO2): available from **2018 onward only** —
  this is why the app doesn't allow years before 2019 (2018 itself is a
  partial year of coverage). There is **no way to get satellite-based NO2
  for, say, 2011** — that data simply doesn't exist.
- **ESA WorldCover** (used for the map's visual LULC reference layer) only
  has **2020 and 2021** editions. It is a *static* reference layer on the
  map — it will look the same regardless of which year you select. The
  actual year-by-year LULC percentages come from our own trained model
  applied to each year's Sentinel-2 composite, not from this overlay.
- The model was trained on **8 regions across India** — treat its accuracy
  as most trustworthy within similar Indian landscapes, not validated
  globally.

## Project structure

```
geocompare/
├── app.py                  # Streamlit UI — single location, multi-year
├── gee_utils.py             # Earth Engine auth, Sentinel-2 + Sentinel-5P (NO2) fetching, ring geometry
├── analysis.py              # RF classification, true-annulus rings, year-loop analysis
├── map_utils.py              # Folium map rendering, legend, ring overlays
├── train_model_colab.py      # Colab script: retrain the RF model with traceable WorldCover labels
├── model.pkl                 # Trained RandomForestClassifier (binary)
├── model_metadata.json       # Training provenance, accuracy, confusion matrix
├── requirements.txt
├── .gitignore
└── README.md
```

## Model performance (update this after retraining)

Current model: **76.88% accuracy** (11 features: 6 spectral bands + NDVI/NDWI/NDBI + BSI + local texture).
Per-class recall: Vegetation 84.6%, Built-up 65%, Water 91.7%, Bare Land 66.25%.
Built-up and Bare Land remain the hardest to distinguish — a known,
explainable limitation of spectral-only classification (see `model_metadata.json`
for the full confusion matrix and notes on what was tried to improve this).

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
   **Service Usage Consumer**, **Earth Engine Resource Viewer**, and
   **Earth Engine Resource Writer** (the last one is required for map tile
   generation via `getMapId`, not just data reads).
3. Make sure `model.pkl` is present in the repo root, and `model_metadata.json`
   reflects its actual training run.
4. Run locally:
   ```bash
   streamlit run app.py
   ```

## Known limitations / roadmap

- **NO2 and LULC use different underlying resolutions** (Sentinel-5P ~1km vs
  Sentinel-2 ~10-30m) — the ring-level averaging smooths over this, but it's
  worth knowing the two signals aren't pixel-for-pixel comparable.
- **Correlation shown is descriptive, not causal** — a ring showing both
  rising Built-up % and rising NO2 suggests a relationship worth
  investigating, not a proven cause-and-effect claim.
- **The LULC map overlay is a static reference layer**, not a rendered
  version of our own model's per-year predictions — see the note under
  Data Sources above for why.
- **Model accuracy ceiling**: further gains beyond ~77% likely need more
  training data, multi-temporal features, or a different model architecture
  (see `model_metadata.json` notes and chat history for a fuller list of
  options considered).

## Contributing / editing on GitHub

- Never commit `.streamlit/secrets.toml` or any real service-account JSON.
- If you retrain `model.pkl`, update `model_metadata.json` and this README's
  "Model performance" section in the same PR — keep them in sync.
- If you change `FEATURE_COLS` in `analysis.py`, you MUST also update
  `train_model_colab.py`'s feature list and `gee_utils.py`'s feature-image
  builder to match, or predictions will break or error out with a
  feature-mismatch.


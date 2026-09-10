# 🌍 GeoCompare

Concentric Land Use / Land Cover (LULC) gradient analysis over Sentinel-2 and
ESA WorldCover, built with Streamlit and Google Earth Engine.

GeoCompare draws **true concentric annuli** (rings) around one or more
coordinates and classifies land cover in each ring separately, so you can see
how a landscape changes from the center outward — e.g. dense built-up core
fading into vegetation at the edges, or vice versa.

## What's new in this version

- **True annulus rings** — each ring (0–1km, 1–2km, 2–3km, etc.) is analyzed
  independently via `outer_buffer.difference(inner_buffer)`, instead of
  nested buffers that double-counted inner pixels in outer-ring stats.
- **Configurable date range & cloud filter** — no longer hardcoded to 2023.
- **Model confidence scores** alongside class percentages (`predict_proba`).
- **Batch mode** — upload a CSV of `name, lat, lon` and analyze many locations
  in one run.
- **Time-series mode** — analyze the same location across multiple years to
  see land-cover change over time.
- **Map legend** for the LULC palette, `st_folium` (interactive, non-deprecated)
  instead of `folium_static`.
- **Robust error handling** — empty Earth Engine collections, missing model
  files, and empty ring samples now surface as clear messages instead of
  silent zero-rows.
- **Input validation** on latitude/longitude bounds.
- **Caching** (`st.cache_resource` / `st.cache_data`) so repeated runs with
  identical inputs don't re-hit Earth Engine.
- **Exports**: CSV for all modes, plus per-map HTML export in Compare mode.

## Project structure

```
geocompare/
├── app.py                 # Streamlit UI (Compare tab + Batch/Time-Series tab)
├── gee_utils.py            # Earth Engine auth + Sentinel-2 feature extraction
├── analysis.py             # RF classification, true-annulus ring stats, batch entry point
├── map_utils.py             # Folium map rendering, legend, ring overlays
├── train_model.py           # Reference script used to produce model.pkl
├── model.pkl                # Trained RandomForestClassifier (binary, not shown here)
├── model_metadata.json      # Fill in after each retrain — keeps predictions traceable
├── requirements.txt
├── .gitignore
└── README.md
```

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
   # ...rest of the standard service account JSON fields
   ```
3. Make sure `model.pkl` is present in the repo root (see `train_model.py` for
   how it was generated, and fill in `model_metadata.json` with training
   provenance so future-you can trust the model's outputs).
4. Run locally:
   ```bash
   streamlit run app.py
   ```

## Known limitations / roadmap

- **Random Forest runs client-side on sampled pixels** pulled via `.getInfo()`,
  which is a network bottleneck at higher `numPixels` or larger areas. A
  future migration to `ee.Classifier.smileRandomForest` (trained and run
  natively inside Earth Engine) would remove this bottleneck and allow
  full-resolution classification instead of sampling.
- **Time-series mode currently summarizes only the outermost ring** per year
  for simplicity; a future version could show the full ring breakdown per
  year (e.g. as a small multiples grid) or compute area-weighted whole-region
  stats.
- **No held-out validation metrics are tracked yet** — add accuracy/confusion
  matrix numbers to `model_metadata.json` next time the model is retrained.
- **Training data provenance for `model.pkl` is not yet documented** — see
  the `FILL_IN_*` placeholders in `model_metadata.json`.

## Contributing / editing on GitHub

- Never commit `.streamlit/secrets.toml` or any real service-account JSON —
  `.gitignore` already excludes them, but double-check before pushing.
- If you retrain `model.pkl`, update `model_metadata.json` in the same PR.
- Feature branches → PR → review is recommended even solo, since Streamlit
  Cloud auto-deploys from your default branch on push.

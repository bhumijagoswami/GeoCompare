"""
analysis.py
Random Forest-based LULC classification and multi-ring spatial statistics.

Changes vs. original:
- Rings are now TRUE ANNULI (outer.difference(inner)) instead of nested buffers,
  so each ring's stats reflect only that zone, not the cumulative area from center.
- Adds mean prediction confidence (via predict_proba) per class per ring.
- Adds a bulk/batch entry point (analyze_locations) for multi-location support.
- Adds defensive checks: missing model file, empty samples, missing feature cols.
- Cached with st.cache_data keyed on all inputs so repeated identical runs
  don't re-sample from Earth Engine.
"""

import os
import joblib
import pandas as pd
import numpy as np
import ee
import streamlit as st

MODEL_PATH = "model.pkl"
FEATURE_COLS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI', 'BSI', 'TEXTURE']
CLASS_NAMES = {0: "Vegetation", 1: "Built-up", 2: "Water", 3: "Bare Land"}


class ModelNotFoundError(Exception):
    pass


class NoSamplesError(Exception):
    pass


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise ModelNotFoundError(
            f"'{MODEL_PATH}' not found. Train it with train_model.py and commit it "
            f"to the repo root (or update MODEL_PATH)."
        )
    return joblib.load(MODEL_PATH)


def _ring_geometry(lat, lon, r_inner, r_outer):
    """True annulus: outer buffer minus inner buffer. r_inner=0 -> solid disk."""
    outer = ee.Geometry.Point([lon, lat]).buffer(r_outer)
    if r_inner <= 0:
        return outer
    inner = ee.Geometry.Point([lon, lat]).buffer(r_inner)
    return outer.difference(inner, ee.ErrorMargin(1))


def _classify_ring(gee_image, geometry, num_pixels=800, scale=30):
    model = load_model()
    samples = gee_image.sample(
        region=geometry,
        scale=scale,
        numPixels=num_pixels,
        geometries=False,
    )
    data_dict = samples.getInfo()
    rows = [f['properties'] for f in data_dict['features']]

    if not rows:
        raise NoSamplesError("No valid pixels sampled in this ring (cloud gaps, water masking, or a tiny ring).")

    df = pd.DataFrame(rows)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise NoSamplesError(f"Sampled data missing expected feature columns: {missing}")

    X = df[FEATURE_COLS]
    preds = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    total = len(preds)
    unique, counts = np.unique(preds, return_counts=True)
    counts_dict = dict(zip(unique, counts))

    pct_stats = {}
    conf_stats = {}
    for class_id, name in CLASS_NAMES.items():
        pct_stats[name] = (counts_dict.get(class_id, 0) / total) * 100
        if probs is not None and class_id < probs.shape[1]:
            # mean confidence the model assigned to this class, across all samples
            conf_stats[name] = float(np.mean(probs[:, class_id])) * 100
        else:
            conf_stats[name] = None

    return pct_stats, conf_stats, total


@st.cache_data(show_spinner=False, ttl=3600)
def analyze_rings(_gee_image, lat, lon, radii, num_pixels=800, scale=30):
    """
    Runs true-annulus ring classification.
    Returns (pct_df_dict, confidence_df_dict, sample_counts_dict).
    Leading underscore on _gee_image tells st.cache_data not to try hashing
    the ee.Image object; lat/lon/radii still form the cache key.
    """
    sorted_radii = sorted(radii)
    pct_results = {}
    conf_results = {}
    counts = {}
    prev_r = 0
    for r in sorted_radii:
        geom = _ring_geometry(lat, lon, prev_r, r)
        label = f"{prev_r}-{r}m"
        try:
            pct_stats, conf_stats, total = _classify_ring(_gee_image, geom, num_pixels, scale)
        except NoSamplesError as e:
            pct_stats = {name: 0.0 for name in CLASS_NAMES.values()}
            conf_stats = {name: None for name in CLASS_NAMES.values()}
            total = 0
        pct_results[label] = pct_stats
        conf_results[label] = conf_stats
        counts[label] = total
        prev_r = r

    return pct_results, conf_results, counts


def analyze_locations(locations, radii, num_pixels=800, scale=30, get_base_image_fn=None,
                       max_radius=None, start_date=None, end_date=None):
    """
    Batch entry point for multi-location support.
    locations: list of dicts like {"name": "A", "lat": .., "lon": ..}
    Returns: dict keyed by location name -> (pct_results, conf_results, counts)
    Requires get_base_image_fn (e.g. gee_utils.get_base_image) to fetch imagery
    per location before ring analysis.
    """
    if get_base_image_fn is None:
        raise ValueError("get_base_image_fn is required for batch analysis")

    out = {}
    for loc in locations:
        img = get_base_image_fn(
            loc["lat"], loc["lon"], max_radius,
            start_date=start_date, end_date=end_date
        )
        out[loc["name"]] = analyze_rings(img, loc["lat"], loc["lon"], radii, num_pixels, scale)
    return out

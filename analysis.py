"""
analysis.py
XGBoost-based LULC classification and multi-ring spatial statistics.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from gee_utils import (
    ring_geometry, get_base_image_for_year, get_no2_for_ring, GEEDataError
)

MODEL_PATH = "model.pkl"
FEATURE_COLS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDWI', 'NDBI', 'BSI', 'TEXTURE', 'NDVI_STD', 'SAR_VV', 'SAR_VH', 'NIGHT_LIGHTS']
CLASS_NAMES = {0: "Vegetation", 1: "Built-up", 2: "Water", 3: "Bare Land"}


class ModelNotFoundError(Exception):
    pass


class NoSamplesError(Exception):
    pass


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise ModelNotFoundError(
            f"'{MODEL_PATH}' not found. Train it with train_model_colab.py and commit it "
            f"to the repo root (or update MODEL_PATH)."
        )
    return joblib.load(MODEL_PATH)


def get_ring_bounds(radii):
    """Turns [2000, 4000, 6000] into [(0,2000), (2000,4000), (4000,6000)]."""
    sorted_radii = sorted(radii)
    bounds = []
    prev_r = 0
    for r in sorted_radii:
        bounds.append((prev_r, r))
        prev_r = r
    return bounds


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
            conf_stats[name] = float(np.mean(probs[:, class_id])) * 100
        else:
            conf_stats[name] = None

    return pct_stats, conf_stats, total


@st.cache_data(show_spinner=False, ttl=3600)
def analyze_rings(_gee_image, lat, lon, radii, num_pixels=800, scale=30):
    pct_results = {}
    conf_results = {}
    counts = {}
    for r_inner, r_outer in get_ring_bounds(radii):
        geom = ring_geometry(lat, lon, r_inner, r_outer)
        label = f"{r_inner}-{r_outer}m"
        try:
            pct_stats, conf_stats, total = _classify_ring(_gee_image, geom, num_pixels, scale)
        except NoSamplesError:
            pct_stats = {name: 0.0 for name in CLASS_NAMES.values()}
            conf_stats = {name: None for name in CLASS_NAMES.values()}
            total = 0
        pct_results[label] = pct_stats
        conf_results[label] = conf_stats
        counts[label] = total

    return pct_results, conf_results, counts


@st.cache_data(show_spinner=False, ttl=3600)
def analyze_location_over_time(lat, lon, radii, years, cloud_filter=10, num_pixels=800, scale=30):
    results = {}
    ring_bounds = get_ring_bounds(radii)

    for year in years:
        year_errors = []

        try:
            img = get_base_image_for_year(lat, lon, max(radii), year, cloud_filter)
            lulc_pct, lulc_conf, lulc_counts = analyze_rings(img, lat, lon, radii, num_pixels, scale)
        except GEEDataError as e:
            year_errors.append(f"LULC: {e}")
            lulc_pct = {f"{i}-{o}m": {name: None for name in CLASS_NAMES.values()} for i, o in ring_bounds}
            lulc_conf = {f"{i}-{o}m": {name: None for name in CLASS_NAMES.values()} for i, o in ring_bounds}
            lulc_counts = {f"{i}-{o}m": 0 for i, o in ring_bounds}

        no2_by_ring = {}
        for r_inner, r_outer in ring_bounds:
            label = f"{r_inner}-{r_outer}m"
            try:
                no2_by_ring[label] = get_no2_for_ring(lat, lon, r_inner, r_outer, year)
            except GEEDataError as e:
                no2_by_ring[label] = None
                year_errors.append(f"NO2 ({label}): {e}")

        results[year] = {
            "lulc_pct": lulc_pct,
            "lulc_conf": lulc_conf,
            "lulc_counts": lulc_counts,
            "no2": no2_by_ring,
            "errors": year_errors,
        }

    return results

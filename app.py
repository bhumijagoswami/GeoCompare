"""
app.py
GeoCompare v3: Single-Location Concentric Ring + Multi-Year Change Analysis

Redesign vs. the earlier A-vs-B comparison app:
- ONE location instead of two, with 3 concentric true-annulus rings.
- Analysis runs across MULTIPLE YEARS (default 2019, 2022, 2026; any custom
  years >= 2019 are allowed, since that's the first full year of Sentinel-5P
  NO2 data used in this app).
- Two change signals per ring per year: LULC composition (from our trained
  Random Forest) and mean NO2 concentration (a raw Sentinel-5P measurement,
  no ML model involved).
- Results are stored in st.session_state so they survive the automatic
  rerun that st_folium triggers on load/interaction (see the comment in the
  run button's try block for why this matters).
"""

import streamlit as st
import pandas as pd
from datetime import date

from gee_utils import (
    get_base_image_for_year, GEEDataError, NO2_MIN_YEAR
)
from analysis import analyze_location_over_time, ModelNotFoundError, CLASS_NAMES, get_ring_bounds
from map_utils import get_map
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="GeoCompare")
st.title("GeoCompare: Concentric Ring Change Analysis Over Time")
st.caption(
    "Pick one location, and see how land use and air quality have changed "
    "over the years - broken down ring by ring, from the center outward."
)

CURRENT_YEAR = date.today().year
DEFAULT_YEARS = [2019, 2022, min(2026, CURRENT_YEAR)]
YEAR_OPTIONS = list(range(NO2_MIN_YEAR, CURRENT_YEAR + 1))

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=28.6139, min_value=-90.0, max_value=90.0, format="%.6f")
with col2:
    lon = st.number_input("Longitude", value=77.2090, min_value=-180.0, max_value=180.0, format="%.6f")

c1, c2 = st.columns(2)
with c1:
    max_radius = st.slider("Max Outer Radius (meters)", 1000, 15000, 6000, step=500)
    radii = [max_radius // 3, (max_radius * 2) // 3, max_radius]
    ring_bounds = get_ring_bounds(radii)
    ring_labels = [f"{i}-{o}m" for i, o in ring_bounds]
    st.info(f"Rings: {', '.join(ring_labels)}")
with c2:
    years = st.multiselect(
        "Years to compare",
        options=YEAR_OPTIONS,
        default=[y for y in DEFAULT_YEARS if y in YEAR_OPTIONS],
        help=f"Sentinel-5P NO2 data starts in {NO2_MIN_YEAR}, so earlier years aren't available.",
    )
    years = sorted(years)

layer = st.selectbox(
    "Map Overlay",
    ["RGB", "NDVI (Vegetation)", "NDWI (Water)", "LULC Classification (2021 reference)"],
    help="RGB/NDVI/NDWI reflect the actual selected year's imagery. The LULC "
         "overlay is a static 2021 reference layer for visual context only.",
)

with st.expander("Advanced: cloud filter"):
    cloud_filter = st.slider("Max cloud cover % (Sentinel-2)", 0, 100, 10)

if len(years) < 2:
    st.warning("Pick at least 2 years to see a meaningful change-over-time comparison.")

run_clicked = st.button("Run Ring x Year Analysis", type="primary", disabled=len(years) < 1)

# ----------------------------------------------------------------------------
# Run analysis
# ----------------------------------------------------------------------------
if run_clicked:
    try:
        with st.spinner(f"Analyzing {len(years)} year(s) x {len(radii)} rings - this can take a minute..."):
            results = analyze_location_over_time(lat, lon, radii, years, cloud_filter)

        # Results (and a couple of things needed to redraw the map) are
        # stashed in session_state because st_folium triggers a Streamlit
        # rerun on load/interaction, which would otherwise wipe out anything
        # computed only inside this `if run_clicked:` block.
        st.session_state["ts_result"] = {
            "lat": lat, "lon": lon, "radii": radii, "years": years, "layer": layer,
            "results": results,
        }
    except ModelNotFoundError as e:
        st.error(f"Model issue: {e}")
        st.session_state.pop("ts_result", None)
    except Exception as e:
        st.error(f"Unexpected error during analysis: {e}")
        st.session_state.pop("ts_result", None)

# ----------------------------------------------------------------------------
# Render (from session_state, so it survives st_folium's rerun)
# ----------------------------------------------------------------------------
state = st.session_state.get("ts_result")
if state:
    lat, lon, radii, years, layer = state["lat"], state["lon"], state["radii"], state["years"], state["layer"]
    results = state["results"]
    ring_bounds = get_ring_bounds(radii)
    ring_labels = [f"{i}-{o}m" for i, o in ring_bounds]

    # Surface any partial errors (e.g. a cloudy year with no clean Sentinel-2
    # scenes, or NO2 unavailable for an old year) without blocking everything else.
    all_errors = []
    for yr, yr_data in results.items():
        for err in yr_data["errors"]:
            all_errors.append(f"{yr}: {err}")
    if all_errors:
        with st.expander(f"Data issues encountered ({len(all_errors)}) - click to view"):
            for err in all_errors:
                st.write(f"- {err}")

    st.write("---")
    st.subheader("Map View by Year")
    st.caption(
        "RGB/NDVI/NDWI show that year's actual satellite composite. The LULC "
        "overlay (if selected) is a static 2021 reference - it won't change "
        "across years; only the ring statistics below do."
    )
    map_cols = st.columns(len(years))
    for idx, yr in enumerate(years):
        with map_cols[idx]:
            st.write(f"**{yr}**")
            try:
                img = get_base_image_for_year(lat, lon, max(radii), yr, 10)
                m = get_map(lat, lon, img, f"Location ({yr})", layer, radii, year=yr)
                st_folium(m, width=None, height=300, key=f"map_{yr}", returned_objects=[])
            except GEEDataError as e:
                st.warning(f"No imagery available for {yr}: {e}")

    # ------------------------------------------------------------------------
    # Build tidy DataFrames for charting: one row per (year, ring), columns
    # for each LULC class % and NO2.
    # ------------------------------------------------------------------------
    rows = []
    for yr in years:
        yr_data = results[yr]
        for label in ring_labels:
            row = {"Year": yr, "Ring": label}
            row.update(yr_data["lulc_pct"].get(label, {}))
            row["NO2 (umol/m2)"] = yr_data["no2"].get(label)
            rows.append(row)
    long_df = pd.DataFrame(rows)

    st.write("---")
    st.subheader("LULC Change Over Time, Per Ring")
    lulc_cols = st.columns(len(ring_labels))
    for idx, label in enumerate(ring_labels):
        with lulc_cols[idx]:
            st.write(f"**Ring {label}**")
            ring_df = long_df[long_df["Ring"] == label].set_index("Year")[list(CLASS_NAMES.values())]
            st.line_chart(ring_df)

    st.write("---")
    st.subheader("NO2 (Air Quality) Change Over Time, Per Ring")
    st.caption(
        "Mean tropospheric NO2 column density (umol/m2) from Sentinel-5P - "
        "higher values generally indicate more traffic/industrial activity. "
        "This is a raw satellite measurement, not a model prediction."
    )
    no2_wide = long_df.pivot(index="Year", columns="Ring", values="NO2 (umol/m2)")
    st.line_chart(no2_wide)

    with st.expander("Show full data table (LULC % + NO2, all years and rings)"):
        st.dataframe(long_df.set_index(["Year", "Ring"]).style.format("{:.2f}"))

    # ------------------------------------------------------------------------
    # Headline comparison: earliest vs. latest selected year
    # ------------------------------------------------------------------------
    if len(years) >= 2:
        st.write("---")
        st.subheader(f"Overall Change: {years[0]} to {years[-1]}")
        change_rows = []
        for label in ring_labels:
            first = long_df[(long_df["Year"] == years[0]) & (long_df["Ring"] == label)].iloc[0]
            last = long_df[(long_df["Year"] == years[-1]) & (long_df["Ring"] == label)].iloc[0]
            row = {"Ring": label}
            for cls in CLASS_NAMES.values():
                if pd.notna(first.get(cls)) and pd.notna(last.get(cls)):
                    row[f"{cls} Delta%"] = last[cls] - first[cls]
                else:
                    row[f"{cls} Delta%"] = None
            if pd.notna(first.get("NO2 (umol/m2)")) and pd.notna(last.get("NO2 (umol/m2)")):
                row["NO2 Delta (umol/m2)"] = last["NO2 (umol/m2)"] - first["NO2 (umol/m2)"]
            else:
                row["NO2 Delta (umol/m2)"] = None
            change_rows.append(row)
        change_df = pd.DataFrame(change_rows).set_index("Ring")
        st.dataframe(change_df.style.format("{:+.2f}"))

        st.caption(
            "A ring where Built-up Delta% is strongly positive and NO2 Delta "
            "is also positive suggests urbanization tracking with worsening "
            "air quality in that zone - the core hypothesis this tool is "
            "built to explore. This is a descriptive correlation, not a "
            "statistically tested causal claim."
        )

    st.write("---")
    st.subheader("Export")
    st.download_button(
        "Download Full Analysis (CSV)",
        long_df.to_csv(index=False),
        "geocompare_timeseries.csv",
        "text/csv",
    )

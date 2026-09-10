"""
app.py
GeoCompare: Concentric LULC Gradient Analysis — Streamlit frontend.

Changes vs. original:
- Input validation on lat/lon (bounds enforced by number_input min/max).
- Configurable date range + cloud-cover threshold (previously hardcoded to 2023).
- True-annulus rings (via analysis.py) with clear "prev-next m" labeling.
- Model confidence displayed alongside class percentages.
- Robust error handling around GEE/model failures (no more silent zero-rows).
- Legend on the map, st_folium instead of deprecated folium_static.
- New "Batch / Time-Series" tab: upload a CSV of locations and/or compare the
  same location across multiple years to see LULC change over time.
- Map export (HTML) alongside the existing CSV export.
"""

import streamlit as st
import pandas as pd
from datetime import date

from gee_utils import get_base_image, GEEDataError
from analysis import analyze_rings, analyze_locations, ModelNotFoundError, CLASS_NAMES
from map_utils import get_map
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="GeoCompare")
st.title("🌍 GeoCompare: Concentric LULC Gradient Analysis")

tab_compare, tab_batch = st.tabs(["📍 Compare A vs B", "📦 Batch / Time-Series"])

# ----------------------------------------------------------------------------
# Shared sidebar-style controls (rendered inline per tab where needed)
# ----------------------------------------------------------------------------

def date_range_controls(key_prefix):
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input(f"Start date", value=date(2023, 1, 1),
                               max_value=date.today(), key=f"{key_prefix}_start")
    with c2:
        end = st.date_input(f"End date", value=date(2023, 12, 31),
                             max_value=date.today(), key=f"{key_prefix}_end")
    with c3:
        cloud = st.slider("Max cloud cover %", 0, 100, 10, key=f"{key_prefix}_cloud")
    if start >= end:
        st.error("Start date must be before end date.")
        st.stop()
    return start.isoformat(), end.isoformat(), cloud


def render_ring_results(pct_results, conf_results, counts, title):
    st.write(f"#### {title}")
    df_pct = pd.DataFrame(pct_results).T
    df_conf = pd.DataFrame(conf_results).T
    zero_sample_rings = [k for k, v in counts.items() if v == 0]
    if zero_sample_rings:
        st.warning(f"No valid samples in ring(s): {', '.join(zero_sample_rings)}. "
                   f"Shown as 0% — likely cloud gaps or a very thin ring.")
    st.line_chart(df_pct)
    with st.expander("Show class % and model confidence table"):
        st.write("**Class composition (%)**")
        st.dataframe(df_pct.style.format("{:.1f}"))
        st.write("**Mean model confidence (%)**")
        st.dataframe(df_conf.style.format("{:.1f}"))
    return df_pct, df_conf


# ----------------------------------------------------------------------------
# TAB 1: Compare A vs B (core workflow, upgraded)
# ----------------------------------------------------------------------------
with tab_compare:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Location A")
        lat1 = st.number_input("Lat A", value=28.4089, min_value=-90.0, max_value=90.0, format="%.6f")
        lon1 = st.number_input("Lon A", value=77.3178, min_value=-180.0, max_value=180.0, format="%.6f")
    with col2:
        st.subheader("Location B")
        lat2 = st.number_input("Lat B", value=26.8439, min_value=-90.0, max_value=90.0, format="%.6f")
        lon2 = st.number_input("Lon B", value=75.5652, min_value=-180.0, max_value=180.0, format="%.6f")

    c1, c2 = st.columns(2)
    with c1:
        max_radius = st.slider("Max Outer Radius (meters)", 1000, 15000, 6000, step=500)
        radii = [max_radius // 3, (max_radius * 2) // 3, max_radius]
        st.info(f"**True annulus rings:** 0-{radii[0]}m, {radii[0]}-{radii[1]}m, {radii[1]}-{radii[2]}m")
    with c2:
        layer = st.selectbox("Map Mask Overlay",
                              ["LULC Classification", "RGB", "NDVI (Vegetation)", "NDWI (Water)"])

    with st.expander("Advanced: date range & cloud filter"):
        start_date, end_date, cloud_filter = date_range_controls("compare")

    if st.button("Run Multi-Ring Analysis", type="primary"):
        try:
            with st.spinner("Fetching imagery for Location A..."):
                img1 = get_base_image(lat1, lon1, max_radius, start_date, end_date, cloud_filter)
            with st.spinner("Fetching imagery for Location B..."):
                img2 = get_base_image(lat2, lon2, max_radius, start_date, end_date, cloud_filter)

            with st.spinner("Classifying rings..."):
                pct1, conf1, counts1 = analyze_rings(img1, lat1, lon1, radii)
                pct2, conf2, counts2 = analyze_rings(img2, lat2, lon2, radii)

        except GEEDataError as e:
            st.error(f"Earth Engine data issue: {e}")
            st.stop()
        except ModelNotFoundError as e:
            st.error(f"Model issue: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error during analysis: {e}")
            st.stop()

        st.write("---")
        st.subheader("🗺️ Map View")
        map_col1, map_col2 = st.columns(2)
        with map_col1:
            mapA = get_map(lat1, lon1, img1, "Location A", layer, radii)
            st_folium(mapA, width=500, height=400, key="mapA")
        with map_col2:
            mapB = get_map(lat2, lon2, img2, "Location B", layer, radii)
            st_folium(mapB, width=500, height=400, key="mapB")

        st.write("---")
        st.subheader("📈 LULC Gradient (Center ➔ Outskirts, true annuli)")
        gc1, gc2 = st.columns(2)
        with gc1:
            df1, _ = render_ring_results(pct1, conf1, counts1, "Location A")
        with gc2:
            df2, _ = render_ring_results(pct2, conf2, counts2, "Location B")

        st.subheader("⚠️ Spatial Change Analysis (Innermost vs. Outermost Ring)")
        inner_k, outer_k = df1.index[0], df1.index[-1]
        change1 = df1.loc[outer_k] - df1.loc[inner_k]
        change2 = df2.loc[outer_k] - df2.loc[inner_k]
        change_df = pd.DataFrame({"Loc A Change": change1, "Loc B Change": change2})
        st.dataframe(change_df.style.format("{:+.1f}%"))

        st.subheader("📥 Export")
        export_df = pd.concat([df1.assign(Location="A"), df2.assign(Location="B")])
        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            st.download_button("Download Analysis (CSV)", export_df.to_csv(),
                                "geocompare_rings.csv", "text/csv")
        with exp2:
            st.download_button("Download Map A (HTML)", mapA.get_root().render(),
                                "location_a_map.html", "text/html")
        with exp3:
            st.download_button("Download Map B (HTML)", mapB.get_root().render(),
                                "location_b_map.html", "text/html")

# ----------------------------------------------------------------------------
# TAB 2: Batch / Time-Series
# ----------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Upload a CSV with columns **name, lat, lon** to run ring analysis on many "
        "locations at once, or use the same location across multiple years below "
        "to see LULC change over time."
    )

    mode = st.radio("Mode", ["Batch (multiple locations)", "Time-Series (one location, multiple years)"])

    b_radius = st.slider("Max Outer Radius (meters)", 1000, 15000, 6000, step=500, key="batch_radius")
    b_radii = [b_radius // 3, (b_radius * 2) // 3, b_radius]

    if mode == "Batch (multiple locations)":
        start_date, end_date, cloud_filter = date_range_controls("batch")
        csv_file = st.file_uploader("Upload locations CSV (columns: name, lat, lon)", type=["csv"])

        if csv_file is not None:
            locs_df = pd.read_csv(csv_file)
            required_cols = {"name", "lat", "lon"}
            if not required_cols.issubset(set(locs_df.columns.str.lower())):
                st.error(f"CSV must contain columns: {required_cols}")
                st.stop()
            locs_df.columns = [c.lower() for c in locs_df.columns]
            st.dataframe(locs_df)

            if st.button("Run Batch Analysis"):
                locations = locs_df.to_dict("records")
                results = {}
                progress = st.progress(0.0, text="Starting batch analysis...")
                errors = []
                for i, loc in enumerate(locations):
                    try:
                        img = get_base_image(loc["lat"], loc["lon"], b_radius, start_date, end_date, cloud_filter)
                        pct, conf, counts = analyze_rings(img, loc["lat"], loc["lon"], b_radii)
                        results[loc["name"]] = pct
                    except (GEEDataError, ModelNotFoundError) as e:
                        errors.append(f"{loc['name']}: {e}")
                    progress.progress((i + 1) / len(locations), text=f"Processed {loc['name']}")

                if errors:
                    st.warning("Some locations failed:\n" + "\n".join(errors))

                if results:
                    combined = pd.concat(
                        {name: pd.DataFrame(r) for name, r in results.items()}, axis=0
                    )
                    combined.index.names = ["Location", "Ring"]
                    st.write("### Batch Results")
                    st.dataframe(combined.style.format("{:.1f}"))
                    st.download_button("Download Batch Results (CSV)", combined.to_csv(),
                                        "geocompare_batch.csv", "text/csv")

    else:  # Time-Series mode
        st.write("Analyze how land use has changed at **one location** across multiple date ranges.")
        c1, c2 = st.columns(2)
        with c1:
            ts_lat = st.number_input("Latitude", value=28.4089, min_value=-90.0, max_value=90.0, format="%.6f", key="ts_lat")
        with c2:
            ts_lon = st.number_input("Longitude", value=77.3178, min_value=-180.0, max_value=180.0, format="%.6f", key="ts_lon")

        years_input = st.text_input("Years to compare (comma-separated)", value="2019,2021,2023")
        cloud_filter_ts = st.slider("Max cloud cover %", 0, 100, 10, key="ts_cloud")

        if st.button("Run Time-Series Analysis"):
            try:
                years = [y.strip() for y in years_input.split(",") if y.strip()]
                yearly_results = {}
                progress = st.progress(0.0, text="Starting time-series analysis...")
                for i, yr in enumerate(years):
                    start_date = f"{yr}-01-01"
                    end_date = f"{yr}-12-31"
                    img = get_base_image(ts_lat, ts_lon, b_radius, start_date, end_date, cloud_filter_ts)
                    pct, conf, counts = analyze_rings(img, ts_lat, ts_lon, b_radii)
                    # collapse rings into one overall composition per year (area-weighted would need
                    # per-ring area; here we show outermost ring as the "whole area" summary)
                    outer_key = list(pct.keys())[-1]
                    yearly_results[yr] = pct[outer_key]
                    progress.progress((i + 1) / len(years), text=f"Processed {yr}")

                ts_df = pd.DataFrame(yearly_results).T
                ts_df.index.name = "Year"
                st.write("### LULC Change Over Time (outer ring composition)")
                st.line_chart(ts_df)
                st.dataframe(ts_df.style.format("{:.1f}"))
                st.download_button("Download Time-Series (CSV)", ts_df.to_csv(),
                                    "geocompare_timeseries.csv", "text/csv")
            except GEEDataError as e:
                st.error(f"Earth Engine data issue: {e}")
            except ModelNotFoundError as e:
                st.error(f"Model issue: {e}")

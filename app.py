import streamlit as st
import pandas as pd
from gee_utils import get_base_image
from analysis import analyze_rings
from map_utils import create_multi_ring_map
from streamlit_folium import folium_static

st.set_page_config(layout="wide")
st.title("🌍 GeoCompare: Concentric LULC Gradient Analysis")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Location A")
    lat1 = st.number_input("Lat A", value=28.4089)
    lon1 = st.number_input("Lon A", value=77.3178)
with col2:
    st.subheader("Location B")
    lat2 = st.number_input("Lat B", value=26.8439)
    lon2 = st.number_input("Lon B", value=75.5652)

c1, c2 = st.columns(2)
with c1:
    max_radius = st.slider("Max Outer Radius (meters)", 3000, 15000, 6000, step=3000)
    # Auto-generates 3 distinct study rings (Inner, Middle, Outer)
    radii = [max_radius // 3, (max_radius * 2) // 3, max_radius]
    st.info(f"**Generating 3 Concentric Rings at:** {radii[0]}m, {radii[1]}m, and {radii[2]}m")
with c2:
    layer = st.selectbox("Map Mask Overlay", ["LULC Classification", "RGB", "NDVI (Vegetation)", "NDWI (Water)"])

if st.button("Run Multi-Ring Analysis"):
    with st.spinner("Extracting strictly masked data & tracking LULC changes..."):
        img1 = get_base_image(lat1, lon1, max_radius)
        img2 = get_base_image(lat2, lon2, max_radius)
        
        stats1 = analyze_rings(img1, lat1, lon1, radii)
        stats2 = analyze_rings(img2, lat2, lon2, radii)
        
        st.write("---")
        map_col1, map_col2 = st.columns(2)
        with map_col1:
            mapA = create_multi_ring_map(lat1, lon1, img1, "Location A", layer, radii)
            folium_static(mapA, width=500, height=400)
        with map_col2:
            mapB = create_multi_ring_map(lat2, lon2, img2, "Location B", layer, radii)
            folium_static(mapB, width=500, height=400)
            
        df1 = pd.DataFrame(stats1).T
        df2 = pd.DataFrame(stats2).T
        
        st.subheader("📈 LULC Gradient (Center ➔ Outskirts)")
        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            st.write("### Location A: Land Use Across Rings")
            st.line_chart(df1)
        with chart_c2:
            st.write("### Location B: Land Use Across Rings")
            st.line_chart(df2)
            
        st.subheader("⚠️ Spatial Change Analysis (Inner vs. Outer Ring)")
        inner_k, outer_k = f"{radii[0]}m", f"{radii[-1]}m"
        change1 = df1.loc[outer_k] - df1.loc[inner_k]
        change2 = df2.loc[outer_k] - df2.loc[inner_k]
        
        change_df = pd.DataFrame({"Loc A Sprawl Change": change1, "Loc B Sprawl Change": change2})
        st.dataframe(change_df.style.format("{:+.1f}%"))
        
        export_df = pd.concat([df1.assign(Location="A"), df2.assign(Location="B")])
        st.download_button("📥 Download Multi-Ring Analysis (CSV)", export_df.to_csv(), "geocompare_rings.csv", "text/csv")

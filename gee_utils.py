import ee
import streamlit as st
from google.oauth2 import service_account

PROJECT_ID = "earth-engine-demo-501808"

def init_gee():
    # 1. Cloud Deployment: Check for Streamlit Secrets FIRST
    if "gcp_service_account" in st.secrets:
        key_dict = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(key_dict)
        ee.Initialize(creds, project=PROJECT_ID)
    else:
        # 2. Local Deployment: Use default auth fallback
        try:
            ee.Initialize(project=PROJECT_ID)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=PROJECT_ID)

def get_base_image(lat, lon, max_radius):
    init_gee()
    poi = ee.Geometry.Point([lon, lat]).buffer(max_radius)
    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poi)
        .filterDate('2023-01-01', '2023-12-31')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    )
    image = dataset.median().clip(poi)
    
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    
    bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11'])
    return bands.addBands([ndvi, ndwi, ndbi])

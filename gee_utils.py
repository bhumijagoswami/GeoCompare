"""
gee_utils.py
Earth Engine initialization and Sentinel-2 + SAR + VIIRS feature extraction.
"""

import ee
import streamlit as st
from google.oauth2 import service_account

DEFAULT_PROJECT_ID = "earth-engine-demo-501808"
CLOUD_FILTER_DEFAULT = 10


class GEEDataError(Exception):
    pass


def _get_project_id() -> str:
    if "gcp_service_account" in st.secrets:
        return st.secrets["gcp_service_account"].get("project_id", DEFAULT_PROJECT_ID)
    return st.secrets.get("gee_project_id", DEFAULT_PROJECT_ID)


@st.cache_resource(show_spinner=False)
def init_gee():
    project_id = _get_project_id()
    if "gcp_service_account" in st.secrets:
        key_dict = dict(st.secrets["gcp_service_account"])
        creds = service_account.Credentials.from_service_account_info(
            key_dict,
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )
        ee.Initialize(creds, project=project_id)
    else:
        try:
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project_id)
    return True


def _build_feature_image(dataset: ee.ImageCollection, poi: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    image = dataset.median().clip(poi)

    def calc_ndvi(img):
        return img.normalizedDifference(['B8', 'B4']).rename('NDVI_STD')
    ndvi_std = dataset.map(calc_ndvi).reduce(ee.Reducer.stdDev()).rename('NDVI_STD')

    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(poi)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .median()
          .select(['VV', 'VH'], ['SAR_VV', 'SAR_VH']))

    viirs = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
             .filterBounds(poi)
             .filterDate(start_date, end_date)
             .median()
             .select(['avg_rad'], ['NIGHT_LIGHTS']))

    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

    bsi = image.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {'SWIR1': image.select('B11'), 'RED': image.select('B4'),
         'NIR': image.select('B8'), 'BLUE': image.select('B2')}
    ).rename('BSI')

    texture = image.select('B8').reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(radius=1),
    ).rename('TEXTURE')

    bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    return bands.addBands([ndvi, ndwi, ndbi, bsi, texture, ndvi_std, s1, viirs])


@st.cache_data(show_spinner=False, ttl=3600)
def get_base_image_id(lat, lon, max_radius, start_date, end_date, cloud_filter=CLOUD_FILTER_DEFAULT):
    init_gee()
    poi = ee.Geometry.Point([lon, lat]).buffer(max_radius)
    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
    )
    size = dataset.size().getInfo()
    if size == 0:
        raise GEEDataError(
            f"No Sentinel-2 scenes found for ({lat:.4f}, {lon:.4f}) "
            f"between {start_date} and {end_date} with <{cloud_filter}% cloud cover."
        )
    return size


def get_base_image(lat, lon, max_radius, start_date="2023-01-01", end_date="2023-12-31",
                    cloud_filter=CLOUD_FILTER_DEFAULT) -> ee.Image:
    init_gee()
    get_base_image_id(lat, lon, max_radius, start_date, end_date, cloud_filter)

    poi = ee.Geometry.Point([lon, lat]).buffer(max_radius)
    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
    )
    return _build_feature_image(dataset, poi, start_date, end_date)


def ring_geometry(lat, lon, r_inner, r_outer):
    outer = ee.Geometry.Point([lon, lat]).buffer(r_outer)
    if r_inner <= 0:
        return outer
    inner = ee.Geometry.Point([lon, lat]).buffer(r_inner)
    return outer.difference(inner, ee.ErrorMargin(1))


def get_base_image_for_year(lat, lon, max_radius, year, cloud_filter=CLOUD_FILTER_DEFAULT) -> ee.Image:
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    return get_base_image(lat, lon, max_radius, start_date, end_date, cloud_filter)


NO2_COLLECTION = "COPERNICUS/S5P/OFFL/L3_NO2"
NO2_BAND = "tropospheric_NO2_column_number_density"
NO2_MIN_YEAR = 2019
NO2_SCALE_METERS = 1113


@st.cache_data(show_spinner=False, ttl=3600)
def get_no2_for_ring(lat, lon, r_inner, r_outer, year):
    if year < NO2_MIN_YEAR:
        raise GEEDataError(f"Sentinel-5P NO2 data isn't available before {NO2_MIN_YEAR}.")
    init_gee()
    geom = ring_geometry(lat, lon, r_inner, r_outer)
    start_date, end_date = f"{year}-01-01", f"{year}-12-31"

    collection = (
        ee.ImageCollection(NO2_COLLECTION)
        .select(NO2_BAND)
        .filterDate(start_date, end_date)
        .filterBounds(geom)
    )
    if collection.size().getInfo() == 0:
        raise GEEDataError(f"No Sentinel-5P NO2 scenes found for this ring in {year}.")

    mean_image = collection.mean()
    stats = mean_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=NO2_SCALE_METERS,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    raw_value = stats.get(NO2_BAND)
    if raw_value is None:
        raise GEEDataError(f"NO2 reduction returned no value for this ring in {year}.")

    return raw_value * 1e6

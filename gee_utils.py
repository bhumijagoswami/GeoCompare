"""
gee_utils.py
Earth Engine initialization and Sentinel-2 feature extraction.

Changes vs. original:
- Project ID now read from st.secrets first, falls back to a constant.
- init_gee() is idempotent and safe to call repeatedly (checked via session_state).
- get_base_image() accepts a configurable date range instead of a hardcoded year.
- Raises a clear GEEDataError if the filtered collection is empty, instead of
  silently returning a fully-masked image.
- Wrapped in st.cache_resource / st.cache_data so repeated runs with identical
  inputs don't re-hit the Earth Engine API.
"""

import ee
import streamlit as st
from google.oauth2 import service_account

DEFAULT_PROJECT_ID = "earth-engine-demo-501808"
CLOUD_FILTER_DEFAULT = 10  # percent


class GEEDataError(Exception):
    """Raised when Earth Engine returns no usable imagery for the given inputs."""
    pass


def _get_project_id() -> str:
    if "gcp_service_account" in st.secrets:
        return st.secrets["gcp_service_account"].get("project_id", DEFAULT_PROJECT_ID)
    return st.secrets.get("gee_project_id", DEFAULT_PROJECT_ID)


@st.cache_resource(show_spinner=False)
def init_gee():
    """Initialize Earth Engine once per session. Cached so it only runs once."""
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


def _build_feature_image(dataset: ee.ImageCollection, poi: ee.Geometry) -> ee.Image:
    image = dataset.median().clip(poi)

    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

    # Bare Soil Index: helps separate dry bare soil from built-up surfaces,
    # which are the two classes our classifier confuses most often.
    bsi = image.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {
            'SWIR1': image.select('B11'),
            'RED': image.select('B4'),
            'NIR': image.select('B8'),
            'BLUE': image.select('B2'),
        }
    ).rename('BSI')

    # Local texture: built-up areas have "busy" irregular patterns (buildings,
    # roads, gaps), while bare land is spatially smooth and uniform. A local
    # standard deviation over a small neighborhood captures that difference,
    # which pure per-pixel spectral values cannot.
    texture = image.select('B8').reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(radius=1),
    ).rename('TEXTURE')

    bands = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
    return bands.addBands([ndvi, ndwi, ndbi, bsi, texture])


@st.cache_data(show_spinner=False, ttl=3600)
def get_base_image_id(lat, lon, max_radius, start_date, end_date, cloud_filter=CLOUD_FILTER_DEFAULT):
    """
    Returns a serializable map ID dict for the composite image, and also
    validates that the collection isn't empty. Cached on all inputs.

    We can't cache an ee.Image object directly (not picklable in the way
    st.cache_data expects across reruns in some environments), so callers
    that need the live ee.Image should call get_base_image() below; this
    function exists for cache-friendly validation and reuse of the size check.
    """
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
            f"between {start_date} and {end_date} with <{cloud_filter}% cloud cover. "
            f"Try widening the date range or raising the cloud threshold."
        )
    return size


def get_base_image(lat, lon, max_radius, start_date="2023-01-01", end_date="2023-12-31",
                    cloud_filter=CLOUD_FILTER_DEFAULT) -> ee.Image:
    """
    Builds the Sentinel-2 feature composite (RGB + NIR/SWIR + NDVI/NDWI/NDBI)
    clipped to a buffer of max_radius around (lat, lon).

    Raises GEEDataError if no imagery is available for the given filters.
    """
    init_gee()
    # Validate + leverage cache for the expensive size() check
    get_base_image_id(lat, lon, max_radius, start_date, end_date, cloud_filter)

    poi = ee.Geometry.Point([lon, lat]).buffer(max_radius)
    dataset = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(poi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
    )
    return _build_feature_image(dataset, poi)

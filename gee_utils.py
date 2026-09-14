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
    Builds the Sentinel-2 feature composite (RGB + NIR/SWIR + NDVI/NDWI/NDBI/BSI/TEXTURE)
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


def ring_geometry(lat, lon, r_inner, r_outer):
    """
    True annulus: outer buffer minus inner buffer. r_inner<=0 -> solid disk.
    Lives here (not in analysis.py) so both LULC classification and NO2
    fetching use the exact same ring shapes.
    """
    outer = ee.Geometry.Point([lon, lat]).buffer(r_outer)
    if r_inner <= 0:
        return outer
    inner = ee.Geometry.Point([lon, lat]).buffer(r_inner)
    return outer.difference(inner, ee.ErrorMargin(1))


def get_base_image_for_year(lat, lon, max_radius, year, cloud_filter=CLOUD_FILTER_DEFAULT) -> ee.Image:
    """Convenience wrapper: builds the Sentinel-2 feature composite for a full calendar year."""
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    return get_base_image(lat, lon, max_radius, start_date, end_date, cloud_filter)


# --- NO2 (air quality) via Sentinel-5P TROPOMI ---------------------------
# Sentinel-5P data only exists from mid-2018 onward. Earlier years will
# raise GEEDataError below.
NO2_COLLECTION = "COPERNICUS/S5P/OFFL/L3_NO2"
NO2_BAND = "tropospheric_NO2_column_number_density"
NO2_MIN_YEAR = 2019  # 2018 exists but only partial-year; 2019 is the first full year
NO2_SCALE_METERS = 1113  # native-ish resolution for S5P L3 gridded products


@st.cache_data(show_spinner=False, ttl=3600)
def get_no2_for_ring(lat, lon, r_inner, r_outer, year):
    """
    Returns the mean tropospheric NO2 column density (mol/m^2, converted to
    µmol/m^2 for readability) over a ring for a given calendar year.

    This is a raw physical measurement, not a model prediction — no ML
    classifier is involved here, unlike the LULC side of the analysis.
    """
    if year < NO2_MIN_YEAR:
        raise GEEDataError(
            f"Sentinel-5P NO2 data isn't available before {NO2_MIN_YEAR} "
            f"(requested {year}). Pick a later year for air quality analysis."
        )
    init_gee()
    geom = ring_geometry(lat, lon, r_inner, r_outer)
    start_date, end_date = f"{year}-01-01", f"{year}-12-31"

    collection = (
        ee.ImageCollection(NO2_COLLECTION)
        .select(NO2_BAND)
        .filterDate(start_date, end_date)
        .filterBounds(geom)
    )
    size = collection.size().getInfo()
    if size == 0:
        raise GEEDataError(
            f"No Sentinel-5P NO2 scenes found for this ring in {year}. "
            f"Try a different year."
        )

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
        raise GEEDataError(f"NO2 reduction returned no value for this ring in {year} (likely too small a ring).")

    # Convert mol/m^2 -> µmol/m^2 (multiply by 1e6) purely for human-readable
    # numbers; typical values then land roughly in the 10-100 range instead
    # of ~0.00001-0.0001.
    return raw_value * 1e6

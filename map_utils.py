"""
map_utils.py
Folium map generation: base imagery overlay, LULC mask, concentric ring markers.

Changes vs. original:
- Adds an HTML legend for the LULC classification palette (previously the
  colors existed but were never explained on the map itself).
- Ring circles are now labeled with both bounds (e.g. "2000-4000m") to match
  the true-annulus analysis in analysis.py.
- Function renamed get_map (create_multi_ring_map kept as an alias for
  backwards compatibility with existing calls).
- No longer monkey-patches folium.Map globally on import; add_ee_layer is a
  plain function now, called explicitly. Avoids surprising side effects if
  this module is imported alongside other folium-using code.
"""

import folium
import ee

LULC_LEGEND = {
    "Vegetation": "#2ca02c",
    "Built-up": "#d62728",
    "Water": "#1f77b4",
    "Bare Land": "#7f7f7f",
}

RING_COLORS = ['#ff0000', '#ffaa00', '#ffff00', '#00ffaa', '#aa00ff']


def _add_ee_layer(m: folium.Map, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True,
    ).add_to(m)


def _add_legend(m: folium.Map, layer_type: str):
    if layer_type != "LULC Classification":
        return
    items = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:2px;color:#1a1a1a;">'
        f'<span style="background:{color};width:12px;height:12px;display:inline-block;'
        f'margin-right:6px;border:1px solid #333;"></span>{name}</div>'
        for name, color in LULC_LEGEND.items()
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: #ffffff; color: #1a1a1a; padding: 10px 12px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 12px;
                font-family: -apple-system, Arial, sans-serif;">
        <b style="color:#1a1a1a;">LULC Classes</b><br>{items}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def get_map(lat, lon, img, popup_name, layer_type, radii):
    m = folium.Map(location=[lat, lon], zoom_start=12)
    max_r = max(radii)
    poi = ee.Geometry.Point([lon, lat]).buffer(max_r)

    if layer_type == "RGB":
        vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4}
        _add_ee_layer(m, img, vis, layer_type)
    elif layer_type == "NDVI (Vegetation)":
        vis = {'bands': ['NDVI'], 'min': 0, 'max': 1, 'palette': ['white', 'green']}
        _add_ee_layer(m, img, vis, layer_type)
    elif layer_type == "NDWI (Water)":
        vis = {'bands': ['NDWI'], 'min': 0, 'max': 1, 'palette': ['white', 'blue']}
        _add_ee_layer(m, img, vis, layer_type)
    elif layer_type == "LULC Classification":
        worldcover = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(poi)
        remapped = worldcover.remap([10, 40, 50, 80, 60], [0, 0, 1, 2, 3], defaultValue=255)
        vis = {'min': 0, 'max': 3, 'palette': list(LULC_LEGEND.values())}
        _add_ee_layer(m, remapped, vis, layer_type)

    folium.Marker([lat, lon], popup=popup_name, tooltip=popup_name).add_to(m)

    sorted_radii = sorted(radii, reverse=True)
    prev_r = 0
    ring_bounds = []
    for r in sorted(radii):
        ring_bounds.append((prev_r, r))
        prev_r = r

    for i, (r_in, r_out) in enumerate(reversed(ring_bounds)):
        color = RING_COLORS[i % len(RING_COLORS)]
        folium.Circle(
            location=[lat, lon],
            radius=r_out,
            color=color,
            weight=3,
            fill=False,
            tooltip=f"Ring: {r_in}-{r_out}m",
        ).add_to(m)

    _add_legend(m, layer_type)
    folium.LayerControl().add_to(m)
    return m


# Backwards-compatible alias
def create_multi_ring_map(lat, lon, img, popup_name, layer_type, radii):
    return get_map(lat, lon, img, popup_name, layer_type, radii)

import folium
import ee

def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer

def create_multi_ring_map(lat, lon, img, popup_name, layer_type, radii):
    m = folium.Map(location=[lat, lon], zoom_start=12)
    max_r = max(radii)
    poi = ee.Geometry.Point([lon, lat]).buffer(max_r)
    
    if layer_type == "RGB":
        vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4}
        m.add_ee_layer(img, vis, f'{layer_type}')
    elif layer_type == "NDVI (Vegetation)":
        vis = {'bands': ['NDVI'], 'min': 0, 'max': 1, 'palette': ['white', 'green']}
        m.add_ee_layer(img, vis, f'{layer_type}')
    elif layer_type == "NDWI (Water)":
        vis = {'bands': ['NDWI'], 'min': 0, 'max': 1, 'palette': ['white', 'blue']}
        m.add_ee_layer(img, vis, f'{layer_type}')
    elif layer_type == "LULC Classification":
        # Pull and strict-clip the ESA data so it only shows inside the circle
        worldcover = ee.Image('ESA/WorldCover/v200/2021').select('Map').clip(poi)
        remapped = worldcover.remap([10, 40, 50, 80, 60], [0, 0, 1, 2, 3], defaultValue=255)
        vis = {'min': 0, 'max': 3, 'palette': ['#2ca02c', '#d62728', '#1f77b4', '#7f7f7f']}
        m.add_ee_layer(remapped, vis, f'{layer_type}')
        
    folium.Marker([lat, lon], popup=popup_name).add_to(m)
    
    # Draw 3 concentric boundary rings
    colors = ['#ff0000', '#ffaa00', '#ffff00'] 
    for i, r in enumerate(sorted(radii, reverse=True)):
        folium.Circle(
            location=[lat, lon],
            radius=r,
            color=colors[i % len(colors)],
            weight=3,
            fill=False,
            tooltip=f"Ring: {r}m"
        ).add_to(m)
        
    return m

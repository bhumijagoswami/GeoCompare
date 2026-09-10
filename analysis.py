import joblib
import pandas as pd
import numpy as np
import ee

model = joblib.load("model.pkl")
CLASS_NAMES = {0: "Vegetation", 1: "Built-up", 2: "Water", 3: "Bare Land"}

def analyze_rings(gee_image, lat, lon, radii):
    results = {}
    for r in sorted(radii):
        poi = ee.Geometry.Point([lon, lat]).buffer(r)
        samples = gee_image.sample(
            region=poi,
            scale=30,
            numPixels=800,
            geometries=False
        )
        
        data_dict = samples.getInfo()
        rows = [f['properties'] for f in data_dict['features']]
        
        if not rows:
            results[f"{r}m"] = {name: 0.0 for name in CLASS_NAMES.values()}
            continue
            
        df = pd.DataFrame(rows)
        feature_cols = ['B2', 'B3', 'B4', 'B8', 'B11', 'NDVI', 'NDWI', 'NDBI']
        preds = model.predict(df[feature_cols])
        
        total = len(preds)
        unique, counts = np.unique(preds, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        
        stats = {}
        for class_id, name in CLASS_NAMES.items():
            pct = (counts_dict.get(class_id, 0) / total) * 100
            stats[name] = pct
        results[f"{r}m"] = stats
        
    return results

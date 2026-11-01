"""Pre-fire vegetation indices and burn prediction (ELG 5163).

Modules, in pipeline order:
    effis      - download EFFIS burnt-area perimeters and select fires
    cdse       - CDSE credentials, OAuth token and Sentinel Hub requests
    sentinel2  - fetch pre-fire Sentinel-2 scenes and prior-year baseline scenes
    raster     - GeoTIFF reading, cloud masking, rasterizing perimeters
    indices    - NDVI, Temporal NDVI and DFI formulas
    dataset    - burned / unburned pixel sampling -> dataset.csv
    bayes      - Bayesian histogram classifier
    evaluate   - leave-one-fire-out validation and metrics
    plots      - figures written to outputs/
"""

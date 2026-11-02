"""Pre-fire vegetation indices and burn prediction (ELG 5163).

Modules, in pipeline order:
    effis      - download EFFIS burnt-area perimeters and select fires
    cdse       - CDSE credentials, OAuth token and Sentinel Hub requests
    sentinel2  - fetch pre-fire, prior-year (baseline) and post-fire Sentinel-2 scenes
    worldcover - ESA WorldCover land cover around each scene (vegetation-only sampling, firebreaks)
    terrain    - Copernicus DEM patches; slope and shaded relief
    raster     - GeoTIFF reading, cloud masking, rasterizing perimeters, remote tiles onto a grid
    indices    - NDVI, Temporal NDVI, DFI, NDMI, NBR and neighbourhood shares
    landscape  - dry clumps, distance to firebreaks, valley/ridge position
    dataset    - per-scene layers and burned / unburned pixel sampling -> dataset.csv
    bayes      - Bayesian histogram classifier and naive Bayes combination
    evaluate   - leave-one-fire-out validation and the result tables
    plots      - figures written to outputs/
    maps       - per-fire maps of flagged vs burned pixels
"""

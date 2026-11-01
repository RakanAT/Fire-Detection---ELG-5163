"""EFFIS burnt-area perimeters: download for a region and select the fires to study."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .config import COUNTRY_CANDIDATES, DATE_CANDIDATES, EFFIS_WFS, ID_CANDIDATES, REGIONS, ROOT
from .utils import find_column


def download_effis(bounds, first_year: int, last_year: int | None, out_path: Path) -> None:
    """Download yearly EFFIS burnt-area layers for a lon/lat box and merge them into one GeoPackage."""
    import geopandas as gpd
    import pandas as pd
    import requests

    caps = requests.get(EFFIS_WFS, params={"service": "WFS", "version": "1.1.0", "request": "GetCapabilities"}, timeout=120)
    caps.raise_for_status()
    available = sorted({int(y) for y in re.findall(r"<Name>modis\.ba\.poly\.(\d{4})</Name>", caps.text)})
    years = [y for y in available if y >= first_year and (last_year is None or y <= last_year)]
    if not years:
        raise SystemExit(f"EFFIS has no yearly burnt-area layers in that range (available: {available}).")

    west, south, east, north = bounds
    cache_dir = out_path.parent / "effis_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for year in years:
        path = cache_dir / f"modis_ba_poly_{year}_{west:.2f}_{south:.2f}_{east:.2f}_{north:.2f}.geojson"
        if not path.exists():
            print(f"Downloading EFFIS burnt areas for {year} ...", flush=True)
            params = {
                "service": "WFS",
                "version": "1.1.0",
                "request": "GetFeature",
                "typename": f"ms:modis.ba.poly.{year}",
                # WFS 1.1.0 with an EPSG URN uses lat/lon axis order.
                "bbox": f"{south},{west},{north},{east},urn:ogc:def:crs:EPSG::4326",
                "outputformat": "application/json; subtype=geojson",
            }
            response = requests.get(EFFIS_WFS, params=params, timeout=900)
            response.raise_for_status()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(response.content)
            tmp.replace(path)
        year_fires = gpd.read_file(path)
        print(f"  {year}: {len(year_fires)} fires", flush=True)
        if not year_fires.empty:
            parts.append(year_fires.set_crs(4326, allow_override=True))

    if not parts:
        raise SystemExit("EFFIS returned no fires for this region and period.")
    gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326).to_file(out_path, driver="GPKG")
    print(f"Saved EFFIS perimeters -> {out_path}")


def select_fires(args: argparse.Namespace) -> None:
    """Clip perimeters to the study region, write all of them to args.fires_file and a sample to args.selected_csv."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    countries = args.country
    if args.region_file:
        region = gpd.read_file(ROOT / args.region_file).to_crs(4326)
        mask = gpd.GeoSeries([region.union_all()], crs=4326)
        region_name = Path(args.region_file).stem
    elif args.region != "none":
        mask = gpd.GeoSeries([box(*REGIONS[args.region]["bbox"])], crs=4326)
        region_name = args.region
        countries = countries or REGIONS[args.region]["countries"]
    else:
        mask, region_name = None, "all"

    if args.effis_shp:
        src = ROOT / args.effis_shp
        if not src.exists():
            raise SystemExit(f"Perimeter file not found: {src}")
    else:
        if mask is None:
            raise SystemExit("Automatic EFFIS download needs a region (--region or --region-file), or pass --effis-shp.")
        first_year = pd.Timestamp(args.start_date).year
        last_year = pd.Timestamp(args.end_date).year if args.end_date else None
        src = ROOT / "data" / f"effis_{region_name}_{first_year}-{last_year or 'latest'}.gpkg"
        if src.exists():
            print(f"Using cached EFFIS perimeters {src}")
        else:
            download_effis(mask.total_bounds, first_year, last_year, src)

    # The mask is applied while reading, so a global file never has to fit in memory in full.
    print(f"\nReading perimeters from {src} (region: {region_name}) ...", flush=True)
    fires = gpd.read_file(src, mask=mask)
    fires = fires[fires.geometry.notna() & ~fires.geometry.is_empty]
    if fires.empty:
        raise SystemExit(f"No perimeters intersect region '{region_name}'.")
    if fires.crs is None:
        raise SystemExit("The perimeter file has no CRS.")

    if countries:
        country_col = args.country_column or find_column(fires.columns, COUNTRY_CANDIDATES)
        if country_col is None:
            if args.country:
                raise SystemExit(f"No country column found; pass --country-column. Columns: {list(fires.columns)}")
            print(f"WARNING: no country column found; '{region_name}' is clipped by bounding box only.")
        else:
            wanted = {c.strip().upper() for c in countries}
            in_box = len(fires)
            fires = fires[fires[country_col].astype(str).str.strip().str.upper().isin(wanted)]
            print(f"Country filter on '{country_col}': kept {len(fires)} of {in_box} fires in the box.")
            if fires.empty:
                raise SystemExit(f"No fires with {country_col} in {sorted(wanted)}; check the values or pass --country.")

    id_col = args.id_column or find_column(fires.columns, ID_CANDIDATES)
    date_col = args.date_column or find_column(fires.columns, DATE_CANDIDATES)
    if id_col is None:
        raise SystemExit(f"No fire ID column found; pass --id-column. Columns: {list(fires.columns)}")

    dates = pd.to_datetime(fires[date_col], errors="coerce") if date_col else pd.Series(pd.NaT, index=fires.index)
    region_fires = gpd.GeoDataFrame(
        {"fire_id": fires[id_col].astype(str).str.strip(), "fire_date": dates.dt.strftime("%Y-%m-%d")},
        geometry=fires.geometry.values,
        crs=fires.crs,
    ).to_crs(4326)
    region_fires["area_ha"] = region_fires.to_crs(6933).area / 1e4  # equal-area projection
    region_fires = region_fires.drop_duplicates("fire_id")

    # Every fire in the region is kept in the file so the build stage can exclude them from unburned rings.
    fires_file = ROOT / args.fires_file
    fires_file.parent.mkdir(parents=True, exist_ok=True)
    region_fires.to_file(fires_file, driver="GPKG")

    if args.fire_id:
        if args.fire_date and len(args.fire_date) != len(args.fire_id):
            raise SystemExit("--fire-date must have the same number of values as --fire-id.")
        selected = region_fires[region_fires["fire_id"].isin(args.fire_id)].set_index("fire_id").reindex(args.fire_id)
        missing = selected.index[selected["geometry"].isna()].tolist()
        if missing:
            raise SystemExit(f"Fire IDs not found in region '{region_name}': {missing}")
        if args.fire_date:
            selected["fire_date"] = args.fire_date
        selected = selected.reset_index()
    else:
        when = pd.to_datetime(region_fires["fire_date"], errors="coerce")
        keep = when.notna() & (when >= pd.Timestamp(args.start_date)) & (region_fires["area_ha"] >= args.min_area_ha)
        if args.end_date:
            keep &= when <= pd.Timestamp(args.end_date)
        selected = region_fires[keep]
        if args.max_fires and len(selected) > args.max_fires:
            selected = selected.sample(n=args.max_fires, random_state=42)

    if selected.empty:
        raise SystemExit("No fires matched the selection filters (dates, --min-area-ha). Relax them or pass --fire-id.")
    if len(selected) < 2:
        print("WARNING: leave-one-fire-out validation needs at least 2 fires.")

    selected_csv = ROOT / args.selected_csv
    selected[["fire_id", "fire_date", "area_ha"]].to_csv(selected_csv, index=False)
    print(f"{len(region_fires)} fires in region '{region_name}' -> {fires_file}")
    print(f"{len(selected)} fires selected ({selected['area_ha'].sum():,.0f} ha total) -> {selected_csv}")

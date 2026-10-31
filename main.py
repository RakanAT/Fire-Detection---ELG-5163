"""Run the full burn-prediction workflow described in README.md.

Stages:
    selftest - selftest_synthetic.py
    select   - clip the perimeter file to the study region (Spain + Portugal by default) and pick fires
    fetch    - 01_fetch_sentinel2.py, once per selected fire
    build    - 02_build_dataset.py
    compare  - 03_bayes_compare.py

Example (EFFIS perimeters for the region are downloaded automatically and cached in data/):
    python main.py --max-fires 20
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["selftest", "select", "fetch", "build", "compare"]

# Approximate lon/lat bounding boxes (W, S, E, N), plus country codes used to drop the parts of
# neighbouring countries a box also covers (needs a country attribute, e.g. EFFIS COUNTRY).
# Pass --region-file with a boundary polygon for an exact clip instead.
REGIONS = {
    # Mainland Spain + Portugal + Balearics; the box also touches southern France, Andorra and Morocco.
    "iberia": {"bbox": (-9.6, 35.9, 4.4, 43.9), "countries": ["ES", "PT", "ESP", "PRT", "SPAIN", "PORTUGAL"]},
    "ontario": {"bbox": (-95.16, 41.68, -74.34, 56.86), "countries": None},
}

# Same exact-name preferences as the stage scripts; override with --id-column / --date-column.
ID_CANDIDATES = ("fire_id", "id", "fireid", "nfireid", "fire_number", "objectid")
DATE_CANDIDATES = ("fire_date", "firedate", "afsdate", "start_date", "startdate", "date")
COUNTRY_CANDIDATES = ("country", "iso2", "iso", "country_code", "cntr_code")

# EFFIS WFS: one burnt-area polygon layer per year (ms:modis.ba.poly.<year>), MODIS-mapped fires >= ~30 ha.
EFFIS_WFS = "https://maps.effis.emergency.copernicus.eu/effis"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pre-fire vegetation index burn-prediction workflow.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES, help="Stages to run (always executed in workflow order).")

    parser.add_argument("--effis-shp", help="Burnt-area perimeter file (any region, e.g. global). If omitted, EFFIS perimeters for the region are downloaded.")
    parser.add_argument("--region", choices=[*REGIONS, "none"], default="iberia", help="Study region used to clip the perimeter file.")
    parser.add_argument("--region-file", help="Optional boundary polygon file for an exact region clip (overrides --region).")
    parser.add_argument("--country", nargs="+", help="Keep only fires with these country values (default: the region's own list, e.g. ES PT).")
    parser.add_argument("--country-column", help="Country attribute in the perimeter file (auto-detected if omitted).")
    parser.add_argument("--id-column", help="Fire ID attribute in the perimeter file (auto-detected if omitted).")
    parser.add_argument("--date-column", help="Fire start date attribute in the perimeter file (auto-detected if omitted).")
    parser.add_argument("--start-date", default="2018-01-01", help="Earliest fire date to use (Sentinel-2 L2A coverage is reliable from 2018).")
    parser.add_argument("--end-date", help="Latest fire date to use.")
    parser.add_argument("--min-area-ha", type=float, default=50.0, help="Skip fires smaller than this (too few pixels after erosion).")
    parser.add_argument("--max-fires", type=int, default=20, help="Random sample of at most this many fires (0 = all).")
    parser.add_argument("--fire-id", nargs="+", default=[], help="Use these fire IDs instead of automatic selection.")
    parser.add_argument("--fire-date", nargs="+", default=[], help="Fire dates (YYYY-MM-DD), one per --fire-id, if the file has no date column.")
    parser.add_argument("--fires-file", default="data/fires_region.gpkg", help="Clipped, standardised perimeter file written by 'select'.")
    parser.add_argument("--selected-csv", default="data/selected_fires.csv", help="Fires chosen by 'select' and fetched by 'fetch'.")

    parser.add_argument("--window-days-before", type=int, default=60)
    parser.add_argument("--window-days-after", type=int, default=10)
    parser.add_argument("--margin-m", type=float, default=500.0, help="Image margin around each perimeter; must exceed --buffer-m.")
    parser.add_argument("--image-dir", default="data/sentinel2", help="Where Sentinel-2 GeoTIFFs are written and read.")

    parser.add_argument("--worldcover-raster", help="Optional WorldCover raster.")
    parser.add_argument("--baseline-dir", help="Optional directory of prior-year NDVI rasters.")
    parser.add_argument("--dataset-csv", default="data/dataset.csv")
    parser.add_argument("--buffer-m", type=float, default=250.0)
    parser.add_argument("--max-samples-per-class", type=int, default=5000)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--index-names", nargs="+", default=["ndvi", "anomaly", "dfi"])

    parser.add_argument("--continue-on-fetch-error", action="store_true", help="Keep going if a single fire fails to download.")
    parser.add_argument("--auth-file", default="config/authentication.env", help="KEY=VALUE file with CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (copy config/authentication.env.example).")
    return parser.parse_args()


def _load_auth_file(path: Path) -> None:
    """Load KEY=VALUE pairs into the environment; variables already set in the shell take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


class _Tee:
    """Write to the console and a log file at the same time."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, text):
        self.console.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()


def _start_log(output_dir: str) -> Path:
    log_dir = ROOT / output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_path


def _run(script: str, *script_args: str) -> None:
    cmd = [sys.executable, str(ROOT / script), *script_args]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    # Pipe the child's output through our (tee'd) stdout so it also lands in the run log.
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    print(f"<<< {script} finished in {time.perf_counter() - start:.1f}s", flush=True)


def _find_column(columns, candidates) -> str | None:
    lowered = {str(c).lower(): c for c in columns}
    return next((lowered[c] for c in candidates if c in lowered), None)


def _download_effis(bounds, first_year: int, last_year: int | None, out_path: Path) -> None:
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


def _stage_select(args: argparse.Namespace) -> None:
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
            _download_effis(mask.total_bounds, first_year, last_year, src)

    # The mask is applied while reading, so a global file never has to fit in memory in full.
    print(f"\nReading perimeters from {src} (region: {region_name}) ...", flush=True)
    fires = gpd.read_file(src, mask=mask)
    fires = fires[fires.geometry.notna() & ~fires.geometry.is_empty]
    if fires.empty:
        raise SystemExit(f"No perimeters intersect region '{region_name}'.")
    if fires.crs is None:
        raise SystemExit("The perimeter file has no CRS.")

    if countries:
        country_col = args.country_column or _find_column(fires.columns, COUNTRY_CANDIDATES)
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

    id_col = args.id_column or _find_column(fires.columns, ID_CANDIDATES)
    date_col = args.date_column or _find_column(fires.columns, DATE_CANDIDATES)
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

    # Every fire in the region is kept in the file so stage 2 can exclude them from unburned rings.
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


def _stage_fetch(args: argparse.Namespace) -> None:
    import pandas as pd

    selected_csv = ROOT / args.selected_csv
    if not selected_csv.exists():
        raise SystemExit(f"{args.selected_csv} not found; run the 'select' stage first.")
    if not (os.getenv("CDSE_CLIENT_ID") and os.getenv("CDSE_CLIENT_SECRET")):
        raise SystemExit(f"Fill in CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in {args.auth_file} (or set them in your shell) before running the fetch stage.")

    selected = pd.read_csv(selected_csv, dtype={"fire_id": str, "fire_date": str})
    failed = []
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        print(f"\n[{i}/{len(selected)}] fire {row.fire_id}", flush=True)
        fetch_args = [
            "--effis-shp", args.fires_file,
            "--fire-id", row.fire_id,
            "--window-days-before", str(args.window_days_before),
            "--window-days-after", str(args.window_days_after),
            "--margin-m", str(args.margin_m),
            "--output-dir", args.image_dir,
        ]
        if isinstance(row.fire_date, str) and row.fire_date:
            fetch_args += ["--fire-date", row.fire_date]
        try:
            _run("01_fetch_sentinel2.py", *fetch_args)
        except subprocess.CalledProcessError:
            if not args.continue_on_fetch_error:
                raise
            failed.append(row.fire_id)
    if failed:
        print(f"WARNING: fetch failed for fire IDs: {', '.join(failed)}")


def _stage_build(args: argparse.Namespace) -> None:
    build_args = [
        "--image-dir", args.image_dir,
        "--effis-shp", args.fires_file,
        "--output-csv", args.dataset_csv,
        "--buffer-m", str(args.buffer_m),
        "--max-samples-per-class", str(args.max_samples_per_class),
    ]
    if args.worldcover_raster:
        build_args += ["--worldcover-raster", args.worldcover_raster]
    if args.baseline_dir:
        build_args += ["--baseline-dir", args.baseline_dir]
    _run("02_build_dataset.py", *build_args)


def _stage_compare(args: argparse.Namespace) -> None:
    _run(
        "03_bayes_compare.py",
        "--dataset-csv", args.dataset_csv,
        "--output-dir", args.output_dir,
        "--index-names", *args.index_names,
    )


def _stage_selftest(args: argparse.Namespace) -> None:
    _run("selftest_synthetic.py")


HANDLERS = {
    "selftest": _stage_selftest,
    "select": _stage_select,
    "fetch": _stage_fetch,
    "build": _stage_build,
    "compare": _stage_compare,
}


def main() -> None:
    args = _parse_args()
    log_path = _start_log(args.output_dir)
    print(f"Run started {time.strftime('%Y-%m-%d %H:%M:%S')}: python {' '.join(sys.argv)}")
    print(f"Log file: {log_path}")
    _load_auth_file(ROOT / args.auth_file)
    if args.margin_m <= args.buffer_m:
        raise SystemExit("--margin-m must be larger than --buffer-m, or the unburned ring is cut off at the image edge.")
    selected = [stage for stage in STAGES if stage in args.stages]
    print(f"Running stages: {', '.join(selected)}")
    try:
        for stage in selected:
            HANDLERS[stage](args)
    except subprocess.CalledProcessError as exc:
        print(f"\nFAILED: {Path(exc.cmd[1]).name} exited with code {exc.returncode}", file=sys.stderr)
        print(f"Log saved to {log_path}")
        sys.exit(exc.returncode)
    print("\nWorkflow complete.")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()

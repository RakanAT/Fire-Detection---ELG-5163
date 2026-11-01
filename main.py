"""Run the burn-prediction workflow described in README.md.

Stages (always executed in this order):
    selftest - synthetic check of the comparison method (only when requested, e.g. --stages selftest)
    select   - clip EFFIS perimeters to the study region (Spain + Portugal by default) and pick fires
    fetch    - download one pre-fire Sentinel-2 scene per selected fire, plus prior-year baseline scenes, from CDSE
    build    - sample burned / unburned pixels and their index values -> data/dataset.csv
    compare  - Bayesian histogram comparison with leave-one-fire-out validation -> outputs/

Example (EFFIS perimeters for the region are downloaded automatically and cached in data/):
    python main.py --max-fires 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback

from firedetect import cdse
from firedetect.config import REGIONS, ROOT
from firedetect.utils import start_log

STAGES = ["selftest", "select", "fetch", "build", "compare"]
DEFAULT_STAGES = ["select", "fetch", "build", "compare"]  # selftest only runs when asked for


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pre-fire vegetation index burn-prediction workflow.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=DEFAULT_STAGES, help="Stages to run (always executed in workflow order; selftest is opt-in).")

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
    parser.add_argument("--image-dir", default="data/sentinel2", help="Where pre-fire Sentinel-2 GeoTIFFs are written and read.")
    parser.add_argument("--baseline-dir", default="data/baseline", help="Where prior-year Sentinel-2 GeoTIFFs for Temporal NDVI are written and read.")
    parser.add_argument("--baseline-years", type=int, default=3, help="Prior years averaged into the Temporal NDVI baseline (0 = no Temporal NDVI).")

    parser.add_argument("--worldcover-raster", help="Optional WorldCover raster.")
    parser.add_argument("--dataset-csv", default="data/dataset.csv")
    parser.add_argument("--buffer-m", type=float, default=250.0)
    parser.add_argument("--erode-m", type=float, default=20.0, help="Inward erosion of the perimeter before sampling burned pixels.")
    parser.add_argument("--max-samples-per-class", type=int, default=5000)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--index-names", nargs="+", default=["ndvi", "temporal_ndvi", "dfi"])

    parser.add_argument("--continue-on-fetch-error", action="store_true", help="Keep going if a single fire fails to download.")
    parser.add_argument("--auth-file", default="config/authentication.env", help="KEY=VALUE file with CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (copy config/authentication.env.example).")
    return parser.parse_args()


def _stage_selftest(args: argparse.Namespace) -> None:
    from tests.selftest_synthetic import run

    run()


def _stage_select(args: argparse.Namespace) -> None:
    from firedetect.effis import select_fires

    select_fires(args)


def _stage_fetch(args: argparse.Namespace) -> None:
    from firedetect.sentinel2 import fetch_selected

    fetch_selected(args.session, args.fires_file, args.selected_csv, args.image_dir, args.baseline_dir, args.baseline_years,
                   args.window_days_before, args.window_days_after, args.margin_m, args.continue_on_fetch_error)


def _stage_build(args: argparse.Namespace) -> None:
    from firedetect.dataset import build_dataset

    baseline_dir = ROOT / args.baseline_dir if args.baseline_years > 0 else None
    path = build_dataset(ROOT / args.image_dir, ROOT / args.fires_file, ROOT / args.dataset_csv, baseline_dir,
                         args.buffer_m, args.erode_m, args.max_samples_per_class, args.worldcover_raster)
    print(f"OK: {path}")


def _stage_compare(args: argparse.Namespace) -> None:
    from firedetect.evaluate import run_comparison

    path = run_comparison(ROOT / args.dataset_csv, ROOT / args.output_dir, args.index_names)
    print(f"OK: {path}")


HANDLERS = {
    "selftest": _stage_selftest,
    "select": _stage_select,
    "fetch": _stage_fetch,
    "build": _stage_build,
    "compare": _stage_compare,
}


def main() -> None:
    args = _parse_args()
    log_path = start_log(ROOT / args.output_dir / "logs")
    # Configured after start_log so log records also reach the run log.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # GDAL warns about the photometric tag of every 7-band GeoTIFF; it is harmless.
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    print(f"Run started {time.strftime('%Y-%m-%d %H:%M:%S')}: python {' '.join(sys.argv)}")
    print(f"Log file: {log_path}")

    cdse.load_auth_file(ROOT / args.auth_file)
    if args.margin_m <= args.buffer_m:
        raise SystemExit("--margin-m must be larger than --buffer-m, or the unburned ring is cut off at the image edge.")
    if args.baseline_years <= 0 and "temporal_ndvi" in args.index_names:
        print("--baseline-years is 0, so Temporal NDVI is left out of the comparison.")
        args.index_names = [name for name in args.index_names if name != "temporal_ndvi"]
    selected = [stage for stage in STAGES if stage in args.stages]
    # Only fetch talks to CDSE, so offline runs (e.g. --stages build compare) skip the check.
    args.session = cdse.connect(args.auth_file) if "fetch" in selected else None
    print(f"Running stages: {', '.join(selected)}")

    for stage in selected:
        print(f"\n=== {stage} ===", flush=True)
        start = time.perf_counter()
        try:
            HANDLERS[stage](args)
        except Exception:
            traceback.print_exc()
            print(f"\nFAILED in stage '{stage}'. Log saved to {log_path}", file=sys.stderr)
            sys.exit(1)
        print(f"--- {stage} finished in {time.perf_counter() - start:.1f}s", flush=True)
    print("\nWorkflow complete.")
    print(f"Log saved to {log_path}")


if __name__ == "__main__":
    main()

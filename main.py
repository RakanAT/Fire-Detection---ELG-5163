"""Run the burn-prediction workflow described in README.md.

Stages (always executed in this order):
    selftest - synthetic check of the comparison method (only when requested, e.g. --stages selftest)
    select   - clip EFFIS perimeters to the study region (Spain + Portugal by default) and pick fires
    fetch    - download pre-fire, prior-year and post-fire Sentinel-2 scenes from CDSE, plus WorldCover and DEM
               patches, for the 10 m main scenes and the wider 20 m context scenes
    build    - sample burned / unburned pixels with their index and landscape values -> data/dataset.csv,
               data/context/dataset.csv
    compare  - Bayesian histogram comparison with leave-one-fire-out validation -> outputs/
    maps     - per-fire maps of flagged vs burned pixels -> outputs/fire_maps/

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
from firedetect.config import (CONTEXT_SCENES, CONTEXT_ZONES, CONTROL_INDEX, MAIN_SCENES, MAIN_ZONES, PRE_FIRE_INDICES,
                               REGIONS, ROOT)
from firedetect.utils import start_log

STAGES = ["selftest", "select", "fetch", "build", "compare", "maps"]
DEFAULT_STAGES = ["select", "fetch", "build", "compare", "maps"]  # selftest only runs when asked for


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
    parser.add_argument("--end-date", default="2025-12-31", help="Latest fire date to use (fixed, so later EFFIS years don't change the analysis).")
    parser.add_argument("--min-area-ha", type=float, default=50.0, help="Skip fires smaller than this (too few pixels after erosion).")
    parser.add_argument("--max-fires", type=int, default=100, help="Random sample of at most this many fires (0 = all).")
    parser.add_argument("--sample-file", default="config/selected_fires.csv", help="Frozen fire sample (committed): 'select' uses these fires so everyone analyses the same ones.")
    parser.add_argument("--new-sample", action="store_true", help="Draw a new random sample instead of using --sample-file, and save it there.")
    parser.add_argument("--excluded-fires", default="config/excluded_fires.csv", help="CSV (fire_id, reason) of fires left out of selection and the dataset.")
    parser.add_argument("--fire-id", nargs="+", default=[], help="Use these fire IDs instead of automatic selection.")
    parser.add_argument("--fire-date", nargs="+", default=[], help="Fire dates (YYYY-MM-DD), one per --fire-id, if the file has no date column.")
    parser.add_argument("--fires-file", default="data/fires_region.gpkg", help="Clipped, standardised perimeter file written by 'select'.")
    parser.add_argument("--selected-csv", default="data/selected_fires.csv", help="Fires chosen by 'select' and fetched by 'fetch'.")

    parser.add_argument("--window-days-before", type=int, default=60)
    parser.add_argument("--window-days-after", type=int, default=10)
    parser.add_argument("--baseline-years", type=int, default=3, help="Prior years averaged into the Temporal NDVI baseline.")
    parser.add_argument("--estimate-only", action="store_true", help="Only print the CDSE processing units the fetch stage still needs, then stop.")

    parser.add_argument("--dataset-csv", default="data/dataset.csv")
    parser.add_argument("--context-dataset-csv", default="data/context/dataset.csv")
    parser.add_argument("--fire-summary-csv", default="data/fire_summary.csv", help="Per-fire summary and pre-burn check written by 'build'.")
    parser.add_argument("--erode-m", type=float, default=20.0, help="Inward erosion of the perimeter before sampling burned pixels.")
    parser.add_argument("--max-samples-per-class", type=int, default=5000, help="Pixels sampled per fire for the burned area and for each unburned zone.")

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--index-names", nargs="+", default=[*PRE_FIRE_INDICES, CONTROL_INDEX],
                        help="Single indices to compare; the naive Bayes combinations are added automatically.")
    parser.add_argument("--map-fires", nargs="+", help="Fire IDs to draw maps for in the 'maps' stage (default: every fire in the dataset).")

    parser.add_argument("--continue-on-fetch-error", action="store_true", help="Keep going if a single fire fails to download.")
    parser.add_argument("--auth-file", default="config/authentication.env", help="KEY=VALUE file with CDSE_CLIENT_ID and CDSE_CLIENT_SECRET (copy config/authentication.env.example).")
    args = parser.parse_args()
    if args.baseline_years < 1:
        parser.error("--baseline-years must be at least 1 (Temporal NDVI needs a baseline).")
    return args


def _stage_selftest(args: argparse.Namespace) -> None:
    from tests.selftest_synthetic import run

    run()


def _stage_select(args: argparse.Namespace) -> None:
    from firedetect.effis import select_fires

    select_fires(args)


def _stage_fetch(args: argparse.Namespace) -> None:
    from firedetect.sentinel2 import fetch_selected

    # (scene set, with a post-fire scene): post-fire scenes are only needed on the main grid.
    scene_sets = [(MAIN_SCENES, True), (CONTEXT_SCENES, False)]
    fetch_selected(args.session, args.fires_file, args.selected_csv, scene_sets, args.baseline_years,
                   args.window_days_before, args.window_days_after, args.continue_on_fetch_error, args.estimate_only,
                   ROOT / args.excluded_fires)


def _stage_build(args: argparse.Namespace) -> None:
    from firedetect.dataset import build_dataset

    common = {"fires_file": ROOT / args.fires_file, "selected_csv": ROOT / args.selected_csv, "excluded_file": ROOT / args.excluded_fires,
              "erode_m": args.erode_m, "max_samples_per_class": args.max_samples_per_class}
    print(f"OK: {build_dataset(MAIN_SCENES, output_csv=ROOT / args.dataset_csv, zones=MAIN_ZONES, summary_csv=ROOT / args.fire_summary_csv, **common)}")
    print("\nContext scenes (20 m, unburned zones near and 1-3 km away):")
    print(f"OK: {build_dataset(CONTEXT_SCENES, output_csv=ROOT / args.context_dataset_csv, zones=CONTEXT_ZONES, **common)}")


def _stage_compare(args: argparse.Namespace) -> None:
    from firedetect.evaluate import run_comparison

    path = run_comparison(ROOT / args.dataset_csv, ROOT / args.context_dataset_csv, ROOT / args.output_dir, args.index_names,
                          ROOT / args.fire_summary_csv)
    print(f"OK: {path}")


def _stage_maps(args: argparse.Namespace) -> None:
    from firedetect.maps import write_fire_maps

    path = write_fire_maps(ROOT / args.dataset_csv, ROOT / args.fires_file, MAIN_SCENES, ROOT / args.output_dir, args.index_names,
                           MAIN_ZONES[0][2], args.map_fires, args.erode_m)
    print(f"OK: {path}")


HANDLERS = {
    "selftest": _stage_selftest,
    "select": _stage_select,
    "fetch": _stage_fetch,
    "build": _stage_build,
    "compare": _stage_compare,
    "maps": _stage_maps,
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
    selected = ["fetch"] if args.estimate_only else [stage for stage in STAGES if stage in args.stages]
    # Only a real fetch talks to CDSE, so offline runs (e.g. --stages build compare) skip the check.
    args.session = cdse.connect(args.auth_file) if "fetch" in selected and not args.estimate_only else None
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

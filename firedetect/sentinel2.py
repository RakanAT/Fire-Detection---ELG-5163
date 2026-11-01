"""Fetch pre-fire Sentinel-2 L2A scenes for the selected fires through the Sentinel Hub Process API.

Each scene covers the fire's bounding box plus a margin, in the fire's UTM zone, at 10 m (coarsened
only when a side would exceed 2500 px). Sentinel Hub builds a least-cloudy mosaic over the date window.

Baseline scenes for Temporal NDVI use the same grid and the same window shifted 1..N years back, so
the prior-year images line up pixel for pixel with the pre-fire image.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .cdse import CdseSession
from .config import BANDS, ID_CANDIDATES, MAX_CLOUD_COVER, MAX_SIDE_PX, NATIVE_RESOLUTION_M, ROOT
from .utils import find_column, parse_date, safe_name

LOGGER = logging.getLogger(__name__)

EVALSCRIPT = f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{bands: {list(BANDS)}, units: "DN"}}],
    output: {{id: "default", bands: {len(BANDS)}, sampleType: "INT16"}}
  }};
}}
function evaluatePixel(s) {{
  return [{", ".join(f"s.{b}" for b in BANDS)}];
}}
"""


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def date_window(fire_date: datetime, days_before: int, days_after: int) -> tuple[str, str]:
    """Pre-fire window: from days_before to days_after days before the fire, as YYYY-MM-DD strings."""
    start = (fire_date - timedelta(days=days_before)).strftime("%Y-%m-%d")
    end = (fire_date - timedelta(days=days_after)).strftime("%Y-%m-%d")
    if end < start:
        raise ValueError("The requested pre-fire window is invalid; check the window size parameters.")
    return start, end


def scene_path(out_dir: Path, fire_id, start: str, end: str) -> Path:
    # Double underscore separates the ID from the dates, so IDs may contain single underscores/dashes.
    return out_dir / f"{safe_name(fire_id)}__{start}_{end}.tif"


def fetch_scene(session: CdseSession, fire: gpd.GeoDataFrame, start: str, end: str, margin_m: float, out_path: Path) -> Path:
    """Download one scene around a fire perimeter (a one-row GeoDataFrame); skip it if it already exists."""
    if out_path.exists():
        LOGGER.info("Already downloaded, skipping: %s", out_path)
        return out_path

    centroid = fire.to_crs(4326).geometry.iloc[0].centroid
    epsg = utm_epsg(centroid.x, centroid.y)
    xmin, ymin, xmax, ymax = fire.to_crs(epsg=epsg).geometry.iloc[0].buffer(margin_m).bounds
    resolution_m = max(NATIVE_RESOLUTION_M, max(xmax - xmin, ymax - ymin) / MAX_SIDE_PX)

    # The output grid follows the bbox CRS (UTM), so resx/resy are in metres.
    request = {
        "input": {
            "bounds": {"bbox": [xmin, ymin, xmax, ymax], "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    # The API needs full ISO-8601 timestamps; the end day is included.
                    "timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                    "mosaickingOrder": "leastCC",
                    "maxCloudCoverage": MAX_CLOUD_COVER,
                },
                "processing": {"upsampling": "BILINEAR", "downsampling": "BILINEAR"},
            }],
        },
        "output": {
            "resx": resolution_m,
            "resy": resolution_m,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    content = session.process(request)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)
    LOGGER.info("Saved Sentinel-2 GeoTIFF to %s (%.1f m/px, %.1f MB)", out_path, resolution_m, len(content) / 1e6)
    return out_path


def fetch_selected(session: CdseSession, fires_file: str, selected_csv: str, image_dir: str, baseline_dir: str,
                   baseline_years: int, days_before: int, days_after: int, margin_m: float, continue_on_error: bool) -> None:
    """Fetch the pre-fire scene and baseline_years prior-year scenes of every fire listed in selected_csv."""
    selected_path = ROOT / selected_csv
    if not selected_path.exists():
        raise SystemExit(f"{selected_csv} not found; run the 'select' stage first.")
    selected = pd.read_csv(selected_path, dtype={"fire_id": str, "fire_date": str})

    perimeters = gpd.read_file(ROOT / fires_file)
    if perimeters.crs is None:
        raise SystemExit(f"{fires_file} has no CRS; cannot locate the fires on the map.")
    id_col = find_column(perimeters.columns, ID_CANDIDATES)
    perimeters["_key"] = perimeters[id_col].astype(str).str.strip()

    failed = []
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        print(f"\n[{i}/{len(selected)}] fire {row.fire_id}", flush=True)
        try:
            fire = perimeters[perimeters["_key"] == str(row.fire_id).strip()].iloc[[0]]
            if not (isinstance(row.fire_date, str) and row.fire_date):
                raise ValueError(f"Fire {row.fire_id} has no date; pass --fire-date.")
            fire_date = parse_date(row.fire_date)
            targets = [(ROOT / image_dir, fire_date)]
            targets += [(ROOT / baseline_dir, fire_date - pd.DateOffset(years=k)) for k in range(1, baseline_years + 1)]
            for out_dir, date in targets:
                start, end = date_window(date, days_before, days_after)
                path = fetch_scene(session, fire, start, end, margin_m, scene_path(out_dir, row.fire_id, start, end))
                print(f"OK: {path}")
        except Exception as exc:
            LOGGER.error("Fetch failed for fire %s: %s", row.fire_id, exc)
            if not continue_on_error:
                raise
            failed.append(row.fire_id)
    if failed:
        print(f"WARNING: fetch failed for fire IDs: {', '.join(failed)}")

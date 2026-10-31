#!/usr/bin/env python3
"""Fetch a pre-fire Sentinel-2 L2A scene for a specific EFFIS fire through CDSE.

The script is intentionally written as a robust, cache-aware pipeline for the
project workflow. It searches the EFFIS shapefile for a fire ID, extracts the
fire geometry and date, builds a pre-fire date window, and then calls the
Sentinel Hub Process API using OAuth2 client credentials from the CDSE service.

It is designed to work with Sentinel-2 L2A scenes and returns a GeoTIFF in a
UTM CRS with a cap of 2500 px per side.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import geopandas as gpd
import numpy as np
import pyproj
import requests
from shapely.geometry import Point

LOGGER = logging.getLogger("fetch_sentinel2")

DEFAULT_BATCH_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch pre-fire Sentinel-2 L2A scenes for an EFFIS fire.")
    parser.add_argument("--effis-shp", required=True, help="Path to the EFFIS burnt-area perimeter shapefile.")
    parser.add_argument("--fire-id", required=True, help="EFFIS fire identifier.")
    parser.add_argument("--fire-date", help="Fire start date in ISO format (YYYY-MM-DD). Optional when a date column exists in the shapefile.")
    parser.add_argument("--window-days-before", type=int, default=60, help="Start of the pre-fire window before the fire date.")
    parser.add_argument("--window-days-after", type=int, default=10, help="End of the pre-fire window before the fire date.")
    parser.add_argument("--output-dir", default="data/sentinel2", help="Where to store downloaded GeoTIFFs.")
    parser.add_argument("--cache-dir", default="data/cache", help="Directory for caching request metadata and intermediate artifacts.")
    parser.add_argument("--min-cloud-cover", type=float, default=100.0, help="Maximum cloud coverage for the query (0 to 100).")
    parser.add_argument("--utm-zone", type=int, help="Optional UTM zone override. If omitted, it is inferred from the fire centroid.")
    parser.add_argument("--margin-m", type=float, default=500.0, help="Margin around the fire perimeter so the unburned ring is inside the image.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return parser.parse_args()


# Exact (case-insensitive) attribute names, in order of preference. Substring matching is avoided
# because e.g. "fire" also matches FIREDATE and "date" also matches LASTUPDATE.
ID_CANDIDATES = ("fire_id", "id", "fireid", "nfireid", "fire_number", "objectid")
DATE_CANDIDATES = ("fire_date", "firedate", "afsdate", "start_date", "startdate", "date")


def _find_matching_column(columns: Iterable[str], *candidates: str) -> str | None:
    columns = list(columns)
    lowered = [str(c).lower() for c in columns]
    for candidate in candidates:
        if candidate.lower() in lowered:
            return columns[lowered.index(candidate.lower())]
    return None


def _safe_name(value) -> str:
    """Filesystem-safe fire ID; 02_build_dataset.py applies the same rule to match images to fires."""
    return re.sub(r"[^A-Za-z0-9.-]+", "-", str(value).strip())


def _identify_fire_row(gdf: gpd.GeoDataFrame, fire_id: str) -> gpd.GeoDataFrame:
    id_column = _find_matching_column(gdf.columns, *ID_CANDIDATES)
    if id_column is None:
        raise ValueError(f"Could not locate a fire-id column (tried {ID_CANDIDATES}) in the EFFIS shapefile.")

    matches = gdf[gdf[id_column].astype(str).str.strip() == str(fire_id).strip()]
    if matches.empty:
        raise ValueError(f"No feature matched fire_id={fire_id!r} in the EFFIS shapefile.")
    return matches.iloc[[0]].copy()


def _resolve_fire_date(gdf_row: gpd.GeoDataFrame, provided_fire_date: str | None) -> datetime:
    if provided_fire_date is not None:
        return datetime.fromisoformat(provided_fire_date)
    date_col = _find_matching_column(gdf_row.columns, *DATE_CANDIDATES)
    if date_col is not None:
        value = gdf_row.iloc[0][date_col]
        if value is not None and str(value).strip() not in {"", "nan", "NaT", "None"}:
            return pd_to_datetime(value)
    raise ValueError("Could not determine the fire date. Provide --fire-date or ensure the shapefile contains a fire-date field.")


def pd_to_datetime(value) -> datetime:
    import pandas as pd

    parsed = pd.to_datetime(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.to_pydatetime()


def _utm_from_point(lon: float, lat: float, fallback_zone: int | None = None) -> int:
    if fallback_zone is not None:
        return fallback_zone
    zone = int((lon + 180) / 6) + 1
    hemisphere = "6" if lat >= 0 else "7"
    return int(f"{zone}{hemisphere}")


def _infer_utm_epsg(lon: float, lat: float, fallback_zone: int | None = None) -> int:
    zone = int((lon + 180) / 6) + 1
    if fallback_zone is not None:
        zone = fallback_zone
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return epsg


def _ensure_cache_dirs(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "requests").mkdir(exist_ok=True)


def get_cdse_oauth_token(client_id: str, client_secret: str) -> str:
    """Obtain an OAuth2 access token from CDSE using client credentials."""
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            response = requests.post(token_url, data=data, headers=headers, timeout=DEFAULT_BATCH_TIMEOUT)
            if response.status_code == 200:
                payload = response.json()
                token = payload.get("access_token")
                if token:
                    return token
                raise ValueError(f"OAuth response did not include an access token: {payload}")
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(retry_after)
                continue
            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"OAuth token request failed with status {response.status_code}: {response.text[:500]}")
        except requests.RequestException as exc:
            if attempt == DEFAULT_MAX_RETRIES:
                raise RuntimeError(f"Could not obtain CDSE token after {attempt} attempts: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError("Unexpected failure while fetching CDSE token.")


def _build_evalscript() -> str:
    return """
    //VERSION=3
    function setup() {
      return {
        input: [{
          bands: ["B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"],
          units: "DN"
        }],
        output: {
          id: "default",
          bands: 7,
          sampleType: "INT16"
        }
      };
    }

    function evaluatePixel(samples) {
      return [
        samples.B03,
        samples.B04,
        samples.B08,
        samples.B11,
        samples.B12,
        samples.SCL,
        samples.dataMask
      ];
    }
    """


def _sentinelhub_process_request(
    token: str,
    bbox_utm: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    epsg_code: int,
    resolution_m: float,
    max_cloud_cover: float,
) -> requests.Response:
    # The output grid follows the bbox CRS (UTM), so resx/resy are in metres.
    req = {
        "input": {
            "bounds": {
                "bbox": list(bbox_utm),
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg_code}"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": start_date, "to": end_date},
                        "mosaickingOrder": "leastCC",
                        "maxCloudCoverage": max_cloud_cover,
                    },
                    "processing": {"upsampling": "BILINEAR", "downsampling": "BILINEAR"},
                }
            ],
        },
        "output": {
            "resx": resolution_m,
            "resy": resolution_m,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": _build_evalscript(),
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            response = requests.post(url, json=req, headers=headers, timeout=DEFAULT_BATCH_TIMEOUT)
            if response.status_code == 200:
                return response
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                time.sleep(retry_after)
                continue
            if response.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Sentinel Hub request failed ({response.status_code}): {response.text[:1000]}")
        except requests.RequestException as exc:
            if attempt == DEFAULT_MAX_RETRIES:
                raise RuntimeError(f"Request failed after {attempt} attempts: {exc}") from exc
            time.sleep(2 ** attempt)
    raise RuntimeError("Unexpected Sentinel Hub failure.")


def _query_fire_scene(fire_row, fire_id, fire_date: datetime, args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if fire_row.crs is None:
        raise ValueError("The EFFIS shapefile has no CRS; cannot locate the fire on the map.")
    centroid = fire_row.to_crs(4326).geometry.iloc[0].centroid
    lon, lat = centroid.x, centroid.y

    epsg_code = _infer_utm_epsg(lon, lat, args.utm_zone)
    filter_start = (fire_date - timedelta(days=args.window_days_before)).strftime("%Y-%m-%d")
    filter_end = (fire_date - timedelta(days=args.window_days_after)).strftime("%Y-%m-%d")

    if filter_end < filter_start:
        raise ValueError("The requested pre-fire window is invalid; check the window size parameters.")

    # Double underscore separates the ID from the dates, so IDs may contain single underscores/dashes.
    out_path = output_dir / f"{_safe_name(fire_id)}__{filter_start}_{filter_end}.tif"
    if out_path.exists():
        LOGGER.info("Already downloaded, skipping: %s", out_path)
        return out_path

    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in your environment before running this script.")

    token = get_cdse_oauth_token(client_id, client_secret)

    fire_utm = fire_row.to_crs(epsg=epsg_code).geometry.iloc[0]
    xmin, ymin, xmax, ymax = fire_utm.buffer(args.margin_m).bounds
    # Native 10 m, coarsened only when the scene would exceed 2500 px per side.
    resolution_m = max(10.0, max(xmax - xmin, ymax - ymin) / 2500.0)

    response = _sentinelhub_process_request(
        token=token,
        bbox_utm=(xmin, ymin, xmax, ymax),
        start_date=filter_start,
        end_date=filter_end,
        epsg_code=epsg_code,
        resolution_m=resolution_m,
        max_cloud_cover=args.min_cloud_cover,
    )

    out_path.write_bytes(response.content)
    LOGGER.info("Saved Sentinel-2 GeoTIFF to %s (%.1f m/px, %.1f MB)", out_path, resolution_m, len(response.content) / 1e6)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()

    if not Path(args.effis_shp).exists():
        raise FileNotFoundError(f"EFFIS shapefile not found: {args.effis_shp}")

    gdf = gpd.read_file(args.effis_shp)
    if gdf.empty:
        raise ValueError("The EFFIS shapefile is empty.")

    fire = _identify_fire_row(gdf, args.fire_id)
    fire_date = _resolve_fire_date(fire, args.fire_date)

    _ensure_cache_dirs(Path(args.cache_dir))
    try:
        output_path = _query_fire_scene(fire, args.fire_id, fire_date, args)
        print(f"OK: {output_path}")
    except Exception as exc:  # pragma: no cover - top-level CLI guard
        LOGGER.error("Fetch failed: %s", exc)
        raise


if __name__ == "__main__":
    main()

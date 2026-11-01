"""Copernicus Data Space Ecosystem (CDSE): credentials, OAuth token and Sentinel Hub Process API requests."""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .config import CDSE_TOKEN_URL, MAX_RETRIES, REQUEST_TIMEOUT_S, SENTINELHUB_PROCESS_URL


def load_auth_file(path: Path) -> None:
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


def _post_with_retries(url: str, **kwargs) -> requests.Response:
    """POST, retrying on rate limits (429), server errors (5xx) and network errors; other statuses are returned."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, timeout=REQUEST_TIMEOUT_S, **kwargs)
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)
            continue
        if response.status_code == 429:
            time.sleep(int(response.headers.get("Retry-After", 2 ** attempt)))
            continue
        if response.status_code >= 500 and attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
            continue
        return response
    return response


class CdseSession:
    """Holds an OAuth token and renews it shortly before it expires."""

    def __init__(self, client_id: str, client_secret: str):
        self._credentials = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
        self._token = None
        self._expires_at = 0.0

    def token(self) -> str:
        if self._token is None or time.time() > self._expires_at - 60:
            response = _post_with_retries(CDSE_TOKEN_URL, data=self._credentials)
            if response.status_code in (400, 401, 403):
                raise PermissionError(f"CDSE rejected the credentials (HTTP {response.status_code}).")
            if not response.ok or "access_token" not in response.json():
                raise RuntimeError(f"Unexpected response from CDSE token service (HTTP {response.status_code}): {response.text[:300]}")
            payload = response.json()
            self._token = payload["access_token"]
            self._expires_at = time.time() + payload.get("expires_in", 600)
        return self._token

    def process(self, request: dict) -> bytes:
        """Send a Sentinel Hub Process API request and return the image bytes."""
        headers = {"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"}
        response = _post_with_retries(SENTINELHUB_PROCESS_URL, json=request, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"Sentinel Hub request failed ({response.status_code}): {response.text[:1000]}")
        return response.content


def connect(auth_file: str) -> CdseSession:
    """Read credentials from the environment and request a token; exit with a clear message if that fails."""
    client_id = os.getenv("CDSE_CLIENT_ID")
    client_secret = os.getenv("CDSE_CLIENT_SECRET")
    if not (client_id and client_secret):
        raise SystemExit(f"Fill in CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in {auth_file} (or set them in your shell) before running the fetch stage.")

    print("Checking CDSE credentials ...", flush=True)
    session = CdseSession(client_id, client_secret)
    try:
        session.token()
    except PermissionError as exc:
        raise SystemExit(f"{exc} Check CDSE_CLIENT_ID and CDSE_CLIENT_SECRET in {auth_file}.")
    except requests.RequestException as exc:
        raise SystemExit(f"Could not reach the CDSE identity service to check credentials: {exc}")
    except RuntimeError as exc:
        raise SystemExit(str(exc))
    print("CDSE credentials OK.")
    return session

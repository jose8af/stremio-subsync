"""Tiny HTTP helpers (stdlib only) with retries."""

from __future__ import annotations

import json
import time
import urllib.request

UA = "stremio-subsync/1.0"


def get(url: str, timeout: float = 20, attempts: int = 3) -> bytes:
    err: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccups are common on the public subtitle mirror
            err = e
            time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {err}")


def get_json(url: str, timeout: float = 20, attempts: int = 3):
    """GET and decode JSON, retrying both network errors and truncated/garbled bodies."""
    err: Exception | None = None
    for i in range(attempts):
        try:
            return json.loads(get(url, timeout=timeout, attempts=1))
        except Exception as e:
            err = e
            time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {err}")

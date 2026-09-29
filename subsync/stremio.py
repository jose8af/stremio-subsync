"""Find the file Stremio is playing on its local streaming server."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from . import config, net

log = logging.getLogger("subsync.stremio")


@dataclass
class Stream:
    url: str
    key: str        # stable id for caching: infohash/file index
    name: str


def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def find_stream(video_size: int | None, filename: str | None, wait: float = 4.0) -> Stream | None:
    """Match the active torrent file by exact size (what Stremio reports) or file name."""
    deadline = time.monotonic() + wait
    while True:
        try:
            stats = net.get_json(f"{config.STREMIO_SERVER}/stats.json?_={time.time()}", timeout=3, attempts=1)
        except Exception as e:
            log.warning("Stremio streaming server not reachable: %s", e)
            stats = {}
        best = None
        for ih, st in (stats or {}).items():
            if not st:
                continue
            for idx, f in enumerate(st.get("files") or []):
                size_ok = video_size and int(f.get("length", -1)) == int(video_size)
                name_ok = filename and _norm(f.get("name", "")) == _norm(filename)
                if size_ok or name_ok:
                    rank = (bool(size_ok) + bool(name_ok), st.get("downloadSpeed") or 0)
                    if best is None or rank > best[0]:
                        best = (rank, Stream(f"{config.STREMIO_SERVER}/{ih}/{idx}", f"{ih}-{idx}", f.get("name", "")))
        if best:
            return best[1]
        if time.monotonic() > deadline:
            return None
        time.sleep(0.5)

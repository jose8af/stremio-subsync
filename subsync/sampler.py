"""Cached, parallel access to speech-probability windows of one video."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import media
from .align import Window, snap
from .vad import SAMPLE_RATE, get_vad

log = logging.getLogger("subsync.sampler")


class Cancelled(Exception):
    pass


class VideoAudio:
    def __init__(self, url: str, workers: int = 6):
        self.url = url
        self.info = media.probe(url)
        if self.info.audio_index is None:
            raise RuntimeError("video has no audio track")
        if self.info.duration <= 0:
            raise RuntimeError("could not read the video duration")
        self.cancelled = threading.Event()
        self.seconds_fetched = 0.0
        self._cache: dict[tuple[int, int], Window | None] = {}
        self._pending: dict[tuple[int, int], threading.Event] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="audio")

    @property
    def duration(self) -> float:
        return self.info.duration

    def max_windows(self, budget_mb: float, window_seconds: float) -> int:
        """How many windows fit in the download budget at this file's bitrate."""
        if self.info.bit_rate <= 0:
            return 10 ** 6
        return int(budget_mb * 1e6 / (self.info.bit_rate / 8 * window_seconds))

    def fetch(self, start: float, length: float, timeout: float = 90) -> Window | None:
        start = snap(start)
        key = (int(round(start * 1000)), int(round(length * 1000)))
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            ev = self._pending.get(key)
            owner = ev is None
            if owner:
                ev = self._pending[key] = threading.Event()
        if not owner:
            ev.wait()
            return self._cache.get(key)
        win = None
        try:
            if self.cancelled.is_set():
                raise Cancelled()
            t = time.monotonic()
            audio = media.extract(self.url, start, length, self.info.audio_index, timeout=timeout)
            win = Window(start, get_vad().speech_probs(audio))
            self.seconds_fetched += len(audio) / SAMPLE_RATE
            log.debug("window %.0fs+%.0fs in %.1fs", start, length, time.monotonic() - t)
        except Cancelled:
            pass
        except Exception as e:  # a failed window is simply skipped by the aligner
            log.warning("audio window at %.0fs failed: %s", start, e)
        with self._lock:
            if win is not None or not self.cancelled.is_set():
                self._cache[key] = win
            self._pending.pop(key, None)
        ev.set()
        return win

    def sample(self, specs: list[tuple[float, float]], timeout: float | None = None) -> list[Window | None]:
        """Decode windows in parallel; a window that takes longer than `timeout` is skipped."""
        out = list(self._pool.map(lambda s: self.fetch(*s, **({"timeout": timeout} if timeout else {})), specs))
        if self.cancelled.is_set():
            raise Cancelled()
        return out

    def close(self) -> None:
        self.cancelled.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

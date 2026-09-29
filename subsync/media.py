"""Probe a (possibly remote) video and pull short audio windows out of it with ffmpeg.

ffmpeg seeks with HTTP range requests, so only the bytes around each window are
downloaded, never the whole file.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from .vad import SAMPLE_RATE

FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


@dataclass
class MediaInfo:
    duration: float
    audio_index: int | None
    bit_rate: float  # bits per second of the whole file (0 if unknown)


def _net_opts(url: str, wait: float = 45) -> list[str]:
    if url.startswith("http"):
        # Microseconds: how long a torrent may take to deliver the pieces we ask for.
        return ["-rw_timeout", str(int(wait * 1e6))]
    return []


def probe(url: str, timeout: float = 60) -> MediaInfo:
    cmd = [FFPROBE, "-v", "error", *_net_opts(url), "-show_entries",
           "format=duration,bit_rate,size:stream=index,codec_type:stream_disposition=default",
           "-of", "json", url]
    res = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
    info = json.loads(res.stdout or b"{}")
    fmt = info.get("format", {})
    duration = float(fmt.get("duration") or 0)
    bit_rate = float(fmt.get("bit_rate") or 0)
    if not bit_rate and duration and fmt.get("size"):
        bit_rate = float(fmt["size"]) * 8 / duration
    audio = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    default = [s for s in audio if s.get("disposition", {}).get("default")]
    pick = (default or audio or [None])[0]
    return MediaInfo(duration=duration, audio_index=pick["index"] if pick else None, bit_rate=bit_rate)


def extract(url: str, start: float, length: float, audio_index: int | None,
            timeout: float = 90) -> np.ndarray:
    """Decode `length` seconds of audio starting at `start` as 16 kHz mono float32."""
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-v", "error", *_net_opts(url, min(45.0, timeout)),
           "-ss", f"{max(0.0, start):.3f}", "-i", url, "-t", f"{length:.3f}",
           "-map", f"0:{audio_index}" if audio_index is not None else "0:a:0",
           "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if res.returncode != 0 and not res.stdout:
        raise RuntimeError(f"ffmpeg failed at {start:.0f}s: {res.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(res.stdout, dtype=np.float32)

"""Settings, overridable with environment variables."""

import os
from pathlib import Path

HOST = os.environ.get("SUBSYNC_HOST", "127.0.0.1")
PORT = int(os.environ.get("SUBSYNC_PORT", "7979"))
PUBLIC_URL = os.environ.get("SUBSYNC_PUBLIC_URL", f"http://127.0.0.1:{PORT}")

# Stremio's built-in streaming server (serves torrents over HTTP with range requests).
STREMIO_SERVER = os.environ.get("SUBSYNC_STREMIO_SERVER", "http://127.0.0.1:11470")

# Where subtitles come from: the same OpenSubtitles addon Stremio ships with.
OPENSUBTITLES_ADDON = os.environ.get("SUBSYNC_OPENSUBTITLES", "https://opensubtitles-v3.strem.io")

# ISO 639-2 codes as used by Stremio/OpenSubtitles (eng, spa, pob, por, fre, ger, ita, ...).
LANGUAGES = [l.strip() for l in os.environ.get("SUBSYNC_LANGUAGES", "spa,eng").split(",") if l.strip()]

# How many different subtitle files to sync per language (they are offered best-first).
CANDIDATES_PER_LANGUAGE = int(os.environ.get("SUBSYNC_CANDIDATES", "3"))

# Parallel ffmpeg readers per video.
AUDIO_WORKERS = int(os.environ.get("SUBSYNC_WORKERS", "10"))

# Subtitles synced at the same time (they share the downloaded audio).
PARALLEL_SYNCS = int(os.environ.get("SUBSYNC_PARALLEL_SYNCS", "4"))

# Roughly how much of the video (MB) the initial audio windows may download. High-bitrate
# files get fewer windows; typical 1-4 GB releases are unaffected.
DOWNLOAD_BUDGET_MB = float(os.environ.get("SUBSYNC_BUDGET_MB", "700"))

# How long a subtitle request may wait for a sync in progress before answering.
SUBTITLE_WAIT_SECONDS = float(os.environ.get("SUBSYNC_WAIT", "90"))

# Syncs scoring below this are considered failures (typical good syncs score 0.5-0.75).
MIN_SCORE = float(os.environ.get("SUBSYNC_MIN_SCORE", "0.3"))

# The first subtitle of a language is handed out as soon as one scores this well, without
# waiting for the other candidates.
EARLY_SCORE = float(os.environ.get("SUBSYNC_EARLY_SCORE", "0.5"))

# macOS notification when every language's SubSync #1 is ready ("0" to turn off).
NOTIFY = os.environ.get("SUBSYNC_NOTIFY", "1") != "0"

CACHE_DIR = Path(os.environ.get("SUBSYNC_CACHE", Path.home() / "Library" / "Caches" / "stremio-subsync"))

"""macOS notifications ("subtitles ready")."""

from __future__ import annotations

import logging
import subprocess

from . import config

log = logging.getLogger("subsync.notify")

_NAMES = {"eng": "English", "spa": "Spanish", "pob": "Portuguese (BR)", "por": "Portuguese", "fre": "French",
          "ger": "German", "ita": "Italian", "dut": "Dutch", "pol": "Polish", "rus": "Russian", "tur": "Turkish",
          "ara": "Arabic", "heb": "Hebrew", "jpn": "Japanese", "kor": "Korean", "chi": "Chinese"}


def language_name(code: str) -> str:
    return _NAMES.get(code, code)


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def send(message: str, subtitle: str = "") -> None:
    if not config.NOTIFY:
        return
    script = f"display notification {_quote(message)} with title \"SubSync\""
    if subtitle:
        script += f" subtitle {_quote(subtitle[:80])}"
    try:
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        log.debug("notification failed: %s", e)

"""Parse and write SRT / WebVTT subtitles."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TS = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)
_TAG = re.compile(r"<[^>]*>|\{[^}]*\}")
_ONLY_SOUND = re.compile(r"^[\s\-]*(\[[^\]]*\]|\([^)]*\)|[♪♫#*\s])+[\s\-]*$")
_AD = re.compile(r"opensubtitles|subtitles by|subs by|sync(ed)? (and|&) correct|www\.|https?://|\.com\b|\.org\b", re.I)


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def is_speech(self) -> bool:
        """Whether this cue is expected to coincide with audible speech."""
        plain = _TAG.sub("", self.text).strip()
        if not plain or _ONLY_SOUND.match(plain) or _AD.search(plain):
            return False
        return "♪" not in plain and "♫" not in plain


def _secs(h, m, s, ms) -> float:
    ms = (ms + "00")[:3]
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def parse(text: str) -> list[Cue]:
    """Parse SRT or VTT text into cues sorted by start time."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip("\n").split("\n")
        for i, line in enumerate(lines):
            m = _TS.search(line)
            if m:
                g = m.groups()
                start, end = _secs(*g[0:4]), _secs(*g[4:8])
                body = "\n".join(lines[i + 1:]).strip()
                if body and end > start:
                    cues.append(Cue(start, end, body))
                break
    cues.sort(key=lambda c: c.start)
    return cues


def _fmt(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    out = []
    n = 0
    for c in cues:
        if c.end <= 0:
            continue
        n += 1
        out.append(f"{n}\n{_fmt(c.start)} --> {_fmt(c.end)}\n{c.text}\n")
    return "\n".join(out)

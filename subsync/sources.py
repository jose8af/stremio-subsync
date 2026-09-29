"""Fetch candidate subtitles from the OpenSubtitles Stremio addon and rank them."""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass

from . import config, net, subtitles

log = logging.getLogger("subsync.sources")

_TOKEN = re.compile(r"[a-z0-9]+")
_SOURCES = {"bluray", "brrip", "bdrip", "web", "webrip", "webdl", "hdtv", "dvdrip", "dvd", "hdrip", "remux"}


@dataclass
class Candidate:
    id: str
    lang: str
    url: str
    release: str
    rank: float = 0.0


def _tokens(s: str) -> set[str]:
    s = s.lower().replace("web-dl", "webdl").replace("web dl", "webdl")
    return set(_TOKEN.findall(s))


def _similarity(filename: str, release: str) -> float:
    a, b = _tokens(filename), _tokens(release)
    if not a or not b:
        return 0.0
    score = len(a & b) / len(a | b)
    group_a = re.findall(r"-([a-z0-9]+)(?:\.[a-z0-9]{2,4})?$", filename.lower())
    if group_a and group_a[0] in b:
        score += 0.5
    if a & b & _SOURCES:
        score += 0.2
    return score


def list_candidates(type_: str, id_: str, extra: dict, filename: str | None) -> list[Candidate]:
    extra_part = ""
    keep = {k: v for k, v in extra.items() if k in ("videoHash", "videoSize", "filename")}
    if keep:
        extra_part = "/" + urllib.parse.urlencode(keep)
    url = f"{config.OPENSUBTITLES_ADDON}/subtitles/{type_}/{urllib.parse.quote(id_)}{extra_part}.json"
    data = net.get_json(url, timeout=20, attempts=4)
    out = []
    for i, s in enumerate(data.get("subtitles", [])):
        release = s.get("movieReleaseName") or s.get("subtitleFileName") or ""
        c = Candidate(id=str(s.get("id")), lang=s.get("lang", ""), url=s["url"], release=release)
        # Keep the addon's own order as a tie-breaker (it lists popular files first).
        c.rank = (_similarity(filename, release) if filename else 0.0) - i * 1e-3
        if s.get("m") == "h":  # matched by file hash: made for this exact release
            c.rank += 2.0
        out.append(c)
    return out


def pick(cands: list[Candidate], lang: str, n: int) -> list[Candidate]:
    mine = sorted((c for c in cands if c.lang == lang), key=lambda c: -c.rank)
    return mine[:n]


def download(c: Candidate) -> list[subtitles.Cue]:
    return subtitles.parse(subtitles.decode(net.get(c.url, timeout=30, attempts=4)))

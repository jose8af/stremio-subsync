"""The Stremio addon: HTTP server exposing a `subtitles` resource.

    python -m subsync.server        # then install http://127.0.0.1:7979/manifest.json in Stremio
"""

from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__, config, notify, stremio
from .jobs import Manager
from .vad import get_vad

log = logging.getLogger("subsync.server")
jobs = Manager()

LANG_CFG = re.compile(r"^[a-z]{3}(,[a-z]{3})*$")


def manifest(langs: list[str]) -> dict:
    return {
        "id": "community.subsync.local",
        "version": __version__,
        "name": "SubSync",
        "description": "Re-times subtitles to the audio of the torrent you're playing, "
                       f"so they stay in sync. Languages: {', '.join(langs)}.",
        "resources": ["subtitles"],
        "types": ["movie", "series"],
        "idPrefixes": ["tt"],
        "catalogs": [],
        "behaviorHints": {"configurable": True, "configurationRequired": False},
    }


def slot_label(rank: int) -> str:
    return "SubSync #1 (best)" if rank == 1 else f"SubSync #{rank}"


def message_srt(text: str) -> str:
    return f"1\n00:00:00,000 --> 00:00:08,000\n{text}\n"


class Handler(BaseHTTPRequestHandler):
    server_version = f"SubSync/{__version__}"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    # ------------------------------------------------------------------ plumbing
    def _headers(self, code: int, ctype: str, length: int | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store")
        if length is None:
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send(self, code: int, body: bytes | str, ctype: str) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self._headers(code, ctype, len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj), "application/json; charset=utf-8")

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        parts = [urllib.parse.unquote(p) for p in path.split("/") if p]
        langs = config.LANGUAGES
        if len(parts) > 1 and LANG_CFG.match(parts[0]) and parts[1] in ("manifest.json", "subtitles", "configure"):
            langs = parts.pop(0).split(",")
        try:
            if not parts or parts == ["configure"]:
                return self._send(200, status_page(langs), "text/html; charset=utf-8")
            if parts == ["manifest.json"]:
                return self._json(manifest(langs))
            if parts == ["status.json"]:
                return self._json([j.as_dict() for j in reversed(jobs.all())])
            if parts[0] == "subtitles" and len(parts) in (3, 4):
                return self._subtitles(parts, langs)
            if parts[0] == "sub" and len(parts) == 4:
                return self._subtitle_file(parts[1], parts[2], parts[3])
            self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ------------------------------------------------------------------ routes
    def _subtitles(self, parts: list[str], langs: list[str]) -> None:
        type_, id_ = parts[1], parts[2]
        extra: dict[str, str] = {}
        if len(parts) == 4:
            extra = dict(urllib.parse.parse_qsl(parts[3]))
        if id_.endswith(".json"):
            id_ = id_[:-5]
        extra = {k: (v[:-5] if v.endswith(".json") else v) for k, v in extra.items()}
        size = int(extra["videoSize"]) if extra.get("videoSize", "").isdigit() else None
        log.info("subtitles request %s %s %s", type_, id_, extra)

        # Stremio may ask before the torrent is registered on its streaming server.
        stream = stremio.find_stream(size, extra.get("filename"), wait=12)
        if not stream:
            log.info("no matching torrent on the Stremio streaming server; nothing to sync")
            return self._json({"subtitles": []})
        job = jobs.start(stream, type_, id_, extra, langs)
        job.listed.wait(12)  # usually ~1 s; lets us list only slots that will be filled

        subs = []
        for lang in langs:
            st = job.langs[lang]
            n = len(st.candidates) if job.listed.is_set() else config.CANDIDATES_PER_LANGUAGE
            for i in range(1, n + 1):
                subs.append({
                    "id": f"subsync-{lang}-{i}",
                    "url": f"{config.PUBLIC_URL}/sub/{stream.key}/{lang}/{i}.srt",
                    "lang": lang,
                    # Shown in Stremio's subtitle menu instead of the language name.
                    "label": slot_label(i),
                })
        self._json({"subtitles": subs, "cacheMaxAge": 0})

    def _subtitle_file(self, key: str, lang: str, name: str) -> None:
        rank = int(re.sub(r"\D", "", name) or 1)
        job = jobs.get(key)
        if not job or lang not in job.langs:
            return self._send(404, message_srt("SubSync: this video is no longer being tracked."),
                              "application/x-subrip; charset=utf-8")
        # Send headers now and hold the body until the sync is done: Stremio's fetcher
        # only times out waiting for headers, not for the body.
        st = job.langs[lang]
        if self._ready(st, rank):
            return self._send(200, self._body(job, lang, rank), "application/x-subrip; charset=utf-8")
        self._headers(200, "application/x-subrip; charset=utf-8")
        t = time.monotonic()
        while not self._ready(st, rank) and time.monotonic() - t < config.SUBTITLE_WAIT_SECONDS:
            st.ready.wait(0.5)
        log.info("%s/%s waited %.1fs", lang, rank, time.monotonic() - t)
        if self.command != "HEAD":
            self.wfile.write(self._body(job, lang, rank).encode("utf-8"))

    @staticmethod
    def _ready(st, rank: int) -> bool:
        return st.ready.is_set() or (rank == 1 and st.pin_best(config.EARLY_SCORE) is not None)

    @staticmethod
    def _body(job, lang: str, rank: int) -> str:
        st = job.langs[lang]
        good = st.ranked(config.MIN_SCORE)
        if good:
            return good[min(rank, len(good)) - 1].srt
        if not st.ready.is_set():
            return message_srt(f"SubSync: still syncing ({job.status}).\nRe-select this subtitle in a moment.")
        return message_srt(f"SubSync: {st.error or 'sync failed'}.")


def _language_cell(code: str, v: dict) -> str:
    name = html.escape(notify.language_name(code))
    if v["watchable_after"] is not None:
        head = f'<span class="ok">Ready to watch</span> <span class="muted">after {v["watchable_after"]:.0f}s</span>'
    elif v["ready"]:
        head = f'<span class="bad">{html.escape(v["error"] or "failed")}</span>'
    else:
        head = '<span class="wait">Syncing…</span>'
    items = "".join(
        f"<li><b>{html.escape(slot_label(i))}</b> <span class=\"score\">{r['score']:.2f}</span> "
        f"<span class=\"muted\">{html.escape(r['release'])}</span></li>"
        for i, r in enumerate(v["results"], 1))
    if v["pending"]:
        items += f'<li class="muted">{v["pending"]} more still syncing</li>'
    if v["rejected"]:
        items += f'<li class="muted">{len(v["rejected"])} rejected (score under {config.MIN_SCORE})</li>'
    return f"<div class=\"lang\"><div><b>{name}</b> · {head}</div><ol>{items}</ol></div>"


def status_page(langs: list[str]) -> str:
    cfg = ",".join(langs)
    manifest_url = f"{config.PUBLIC_URL}/{cfg}/manifest.json"
    rows = []
    for j in reversed(jobs.all()):
        d = j.as_dict()
        res = "".join(_language_cell(l, v) for l, v in d["languages"].items())
        rows.append(f"<tr><td>{html.escape(d['file'])}</td><td>{html.escape(d['status'])}<br>"
                    f"<span class=\"muted\">{d['elapsed']}s · audio {d['audio_seconds']}s of "
                    f"{d['duration'] or '?'}s</span></td><td>{res}</td></tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>SubSync</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5">
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --accent:#6b4eff; --ok:#1a7f37; --bad:#c62828; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141416; --fg:#f2f2f7; --muted:#a1a1a6; --line:#2c2c2e; --accent:#9b87ff; --ok:#3fb950; --bad:#ff6b6b; }} }}
body {{ font: 15px/1.5 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--fg); max-width: 960px; margin: 32px auto; padding: 0 16px; }}
code, input {{ font: 13px ui-monospace, Menlo, monospace; }}
input {{ width: 100%; padding: 8px; border:1px solid var(--line); border-radius:6px; background:transparent; color:var(--fg); box-sizing:border-box; }}
a.btn {{ display:inline-block; margin-top:8px; padding:8px 14px; background:var(--accent); color:#fff; border-radius:6px; text-decoration:none; }}
table {{ width:100%; border-collapse: collapse; margin-top: 16px; }} td, th {{ border-top:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; }}
.muted {{ color: var(--muted); }} ol {{ margin:4px 0 10px; padding-left:0; list-style:none; }}
.ok {{ color: var(--ok); font-weight:600; }} .bad {{ color: var(--bad); font-weight:600; }} .wait {{ color: var(--muted); font-weight:600; }}
.score {{ font: 12px ui-monospace, Menlo, monospace; padding: 1px 5px; border:1px solid var(--line); border-radius:4px; }}
</style></head><body>
<h1>SubSync</h1>
<p class="muted">Local Stremio addon that re-times subtitles to the audio of the torrent you're playing.</p>
<p>Install URL (languages <code>{html.escape(cfg)}</code>; change them in the URL path, e.g. <code>/eng,spa/manifest.json</code>):</p>
<input readonly value="{html.escape(manifest_url)}" onclick="this.select()">
<a class="btn" href="stremio://{html.escape(manifest_url.split('://', 1)[1])}">Install in Stremio</a>
<h2>Recent videos</h2>
<p class="muted">Subtitles are listed in the same order as in Stremio's subtitle menu. Score: how well the subtitle's timing matches the speech in the video (good syncs are usually 0.5–0.75).</p>
<table><tr><th>File</th><th>Status</th><th>Subtitles</th></tr>
{''.join(rows) or '<tr><td colspan=3 class="muted">Nothing yet. Play a torrent in Stremio.</td></tr>'}
</table></body></html>"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=get_vad, daemon=True).start()  # load the model while we wait
    srv = ThreadingHTTPServer((config.HOST, config.PORT), Handler)
    srv.daemon_threads = True
    log.info("SubSync %s on %s  (install %s/manifest.json, languages %s)",
             __version__, config.PUBLIC_URL, config.PUBLIC_URL, ",".join(config.LANGUAGES))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""One background job per video: fetch candidate subtitles, sample the audio, sync, cache."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from . import align, config, notify, sources, subtitles
from .sampler import Cancelled, VideoAudio
from .stremio import Stream

log = logging.getLogger("subsync.jobs")


@dataclass
class Result:
    candidate: sources.Candidate
    score: float
    srt: str
    summary: str
    seconds: float = 0.0


@dataclass
class LangState:
    ready: threading.Event = field(default_factory=threading.Event)
    candidates: list[sources.Candidate] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)
    error: str | None = None
    # Slot 1 may be answered before every candidate is done; once served it stays put.
    pinned: Result | None = None
    # When slot 1 first became available (seconds after the job started).
    watchable_after: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, r: Result, elapsed: float) -> None:
        with self.lock:
            self.results.append(r)
            self.results.sort(key=lambda r: -r.score)
            if self.watchable_after is None and r.score >= config.EARLY_SCORE:
                self.watchable_after = elapsed

    @property
    def settled(self) -> bool:
        """Slot 1 can be served now (or never will be)."""
        return self.watchable_after is not None or self.ready.is_set()

    def ranked(self, min_score: float) -> list[Result]:
        with self.lock:
            good = [r for r in self.results if r.score >= min_score]
            if self.pinned in good:
                good.remove(self.pinned)
                good.insert(0, self.pinned)
            return good

    def pin_best(self, min_score: float) -> Result | None:
        """Pick slot 1 now if some candidate is already good enough."""
        with self.lock:
            if self.pinned is None and self.results and self.results[0].score >= min_score:
                self.pinned = self.results[0]
            return self.pinned


class Job:
    def __init__(self, stream: Stream, type_: str, id_: str, extra: dict, languages: list[str]):
        self.stream, self.type, self.id, self.extra = stream, type_, id_, extra
        self.languages = languages
        self.langs = {l: LangState() for l in languages}
        self.listed = threading.Event()
        self.status = "starting"
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.video: VideoAudio | None = None
        self._notified = False
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"job-{stream.key[:8]}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ public
    def cancel(self) -> None:
        self._cancel.set()
        if self.video:
            self.video.close()

    @property
    def done(self) -> bool:
        return self.finished is not None

    def wait_for(self, lang: str, timeout: float) -> LangState | None:
        st = self.langs.get(lang)
        if st:
            st.ready.wait(timeout)
        return st

    def as_dict(self) -> dict:
        return {
            "key": self.stream.key, "file": self.stream.name, "id": self.id, "status": self.status,
            "error": self.error, "elapsed": round((self.finished or time.time()) - self.started, 1),
            "audio_seconds": round(self.video.seconds_fetched) if self.video else 0,
            "duration": round(self.video.duration) if self.video else None,
            "languages": {
                l: {"ready": st.ready.is_set(), "error": st.error,
                    "watchable_after": None if st.watchable_after is None else round(st.watchable_after, 1),
                    "pending": len(st.candidates) - len(st.results) if not st.ready.is_set() else 0,
                    # Same order as SubSync #1, #2, #3 in Stremio.
                    "results": [{"release": r.candidate.release, "score": round(r.score, 3),
                                 "sync": r.summary, "seconds": round(r.seconds, 1)}
                                for r in st.ranked(config.MIN_SCORE)],
                    "rejected": [{"release": r.candidate.release, "score": round(r.score, 3)}
                                 for r in st.results if r.score < config.MIN_SCORE]}
                for l, st in self.langs.items()},
        }

    # ------------------------------------------------------------------ worker
    def _check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def _run(self) -> None:
        cache_dir = config.CACHE_DIR / self.stream.key
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.status = "finding subtitles"
            with ThreadPoolExecutor(2) as pool:
                listing = pool.submit(sources.list_candidates, self.type, self.id, self.extra,
                                      self.extra.get("filename") or self.stream.name)
                opening = pool.submit(VideoAudio, self.stream.url, config.AUDIO_WORKERS)
                try:
                    cands = listing.result()
                finally:
                    for l in self.languages:
                        self.langs[l].candidates = sources.pick(
                            cands if listing.done() and not listing.exception() else [], l,
                            config.CANDIDATES_PER_LANGUAGE)
                    self.listed.set()
                self.status = "opening video"
                self.video = opening.result()
            if self._cancel.is_set():
                self.video.close()
                raise Cancelled()

            downloads = ThreadPoolExecutor(4)
            fetched = {c.id: downloads.submit(sources.download, c)
                       for l in self.languages for c in self.langs[l].candidates}
            # Sync candidates concurrently (they share the audio windows), best-ranked and
            # first language first, so each language is ready as soon as its slowest one is.
            order = [(l, c) for i in range(config.CANDIDATES_PER_LANGUAGE)
                     for l in self.languages for c in self.langs[l].candidates[i:i + 1]]
            self.status = f"syncing {len(order)} subtitles"
            left = {l: len(self.langs[l].candidates) for l in self.languages}
            for l in self.languages:
                if not left[l]:
                    self._finish_lang(l)
            with ThreadPoolExecutor(config.PARALLEL_SYNCS) as syncs:
                futures = {syncs.submit(self._sync_one, c, fetched[c.id], cache_dir): (l, c) for l, c in order}
                for fut in as_completed(futures):
                    l, c = futures[fut]
                    try:
                        self.langs[l].add(fut.result(), time.time() - self.started)
                    except Cancelled:
                        raise
                    except Exception as e:
                        log.warning("%s: %s failed: %s", l, c.release, e)
                    left[l] -= 1
                    if not left[l]:
                        self._finish_lang(l)
                    self._maybe_notify()
            downloads.shutdown(wait=False)
            self.status = "done"
        except Cancelled:
            self.status = "cancelled"
        except Exception as e:
            log.exception("job failed")
            self.status, self.error = "failed", str(e)
        finally:
            self.listed.set()
            for st in self.langs.values():
                if not st.ready.is_set():
                    st.error = st.error or self.error or self.status
                    st.ready.set()
            if self.video:
                self.video.close()
            self.finished = time.time()
            log.info("job %s %s in %.1fs", self.stream.name, self.status, self.finished - self.started)

    def _finish_lang(self, l: str) -> None:
        st = self.langs[l]
        if not st.candidates:
            st.error = "no subtitles found for this language"
        elif not st.ranked(config.MIN_SCORE):
            st.error = "could not sync any subtitle reliably"
        elif st.watchable_after is None:
            st.watchable_after = time.time() - self.started
        st.ready.set()
        self._maybe_notify()

    def _maybe_notify(self) -> None:
        """Tell the user once every language's SubSync #1 can be picked."""
        if self._notified or not all(st.settled for st in self.langs.values()):
            return
        self._notified = True
        ok = [notify.language_name(l) for l, st in self.langs.items() if st.watchable_after is not None]
        bad = [notify.language_name(l) for l, st in self.langs.items() if st.watchable_after is None]
        took = max((st.watchable_after or 0) for st in self.langs.values())
        if ok:
            msg = f"{' & '.join(ok)} ready after {took:.0f}s: pick SubSync #1."
            if bad:
                msg += f" No reliable {' / '.join(bad)} subtitle."
        else:
            msg = "Couldn't sync any subtitle for this video."
        log.info("ready to watch: %s", msg)
        notify.send(msg, self.stream.name)

    def _sync_one(self, c: sources.Candidate, download, cache_dir) -> Result:
        meta_path, srt_path = cache_dir / f"{c.id}.json", cache_dir / f"{c.id}.srt"
        if meta_path.exists() and srt_path.exists():
            meta = json.loads(meta_path.read_text())
            return Result(c, meta["score"], srt_path.read_text(encoding="utf-8"), meta["summary"])
        t = time.monotonic()
        cues = download.result()
        self._check()
        res = align.sync(cues, self.video.sample, self.video.duration,
                         self.video.max_windows(config.DOWNLOAD_BUDGET_MB, align.WINDOW_SECONDS))
        srt = subtitles.to_srt(res.cues)
        took = time.monotonic() - t
        srt_path.write_text(srt, encoding="utf-8")
        meta_path.write_text(json.dumps({"score": res.score, "summary": res.summary(),
                                         "release": c.release, "lang": c.lang, "seconds": took}))
        log.info("%s %s: %s (%.1fs, %.0fs of audio so far)", c.lang, c.release, res.summary(), took,
                 self.video.seconds_fetched)
        return Result(c, res.score, srt, res.summary(), took)


class Manager:
    """Keeps recent jobs; starting a new video cancels work on the previous one."""

    def __init__(self, keep: int = 10):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._keep = keep

    def start(self, stream: Stream, type_: str, id_: str, extra: dict, languages: list[str]) -> Job:
        with self._lock:
            job = self._jobs.get(stream.key)
            if job and (job.status not in ("cancelled", "failed")) and \
                    set(languages) <= set(job.languages):
                return job
            for other in self._jobs.values():
                if not other.done:
                    other.cancel()
            job = Job(stream, type_, id_, extra, languages)
            self._jobs.pop(stream.key, None)
            self._jobs[stream.key] = job
            while len(self._jobs) > self._keep:
                self._jobs.pop(next(iter(self._jobs)))
            return job

    def get(self, key: str) -> Job | None:
        with self._lock:
            return self._jobs.get(key)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

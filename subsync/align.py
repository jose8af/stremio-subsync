"""Align subtitles to a video's speech using sparse audio samples.

Model: a subtitle time u shows at video time t where

    u / ratio = t + delta

`ratio` absorbs framerate mismatches (23.976 vs 25 fps, ...) and `delta` is
piecewise constant, which covers the usual reasons subtitles drift out of sync: a
different cut, a missing or extra intro/recap, ad breaks, or a PAL speed-up.

We only decode short audio windows scattered through the video, so a sync reads a
small fraction of the file. A single 30 s window is often ambiguous on its own, so
windows are decoded jointly: a Viterbi pass picks one offset per window and pays a
penalty for every change of offset. Where the path jumps, we sample in between
(considering only the two offsets on either side) until the jump is pinned down,
then choose the exact cue to break at from the audio around it.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .subtitles import Cue
from .vad import FRAME_RATE as FR

log = logging.getLogger("subsync.align")

_FPS = (23.976, 24.0, 25.0, 29.97, 30.0)
RATIOS = sorted({round(a / b, 6) for a in _FPS for b in _FPS})

WINDOW_SECONDS = 30.0
SAMPLE_SPACING = 170.0  # one initial window per this many seconds of video
MIN_WINDOWS, MAX_WINDOWS = 10, 40
INITIAL_PHASE = 0.5     # where in each spacing step the initial window sits (0..1)
MAX_OFFSET = 900.0      # widest offset searched (seconds)
# Viterbi costs of changing offset between consecutive windows. Real subtitles mostly
# jump by a few seconds; big mid-video jumps are rarer and look like chance matches,
# so they must be backed by more windows.
STEP_COST = 0.01        # per frame (32 ms) of slow drift
JUMPS = ((10.0, 0.3), (150.0, 0.8))  # (up to seconds, cost)
FAR_JUMP_COST = 1.2     # anything bigger
FINE_RATIO = 0.004      # also search ratios within +-0.4 % of the standard one (slow drift)
SAME_OFFSET = 0.35      # windows within this many seconds belong to the same piece
MASK_VAR_GUARD = 0.001  # skip offsets where the subtitle is ~all speech or ~all silence
MIN_SPEECH_SECONDS = 6.0  # windows with less speech than this get proportionally less say
MIN_PIECE_EVIDENCE = 0.8  # summed margin a piece's windows must show over neighbouring offsets
PROBES_PER_GAP = 3       # probes decoded per unresolved jump per round
GAP_RESOLVED = 24.0     # stop bisecting once a jump is located within this span
MAX_EXTRA_WINDOWS = 40
PROBE_TIMEOUT = 30.0     # seconds an extra window may take; a stalled one is skipped
HOPELESS_SCORE = 0.2    # below this after the first pass, don't spend more audio on a subtitle
HYPOTHESIS_RADIUS = 0.6  # probes only consider offsets this close to their neighbours'


@dataclass
class Window:
    start: float            # video seconds, snapped to the frame grid
    probs: np.ndarray       # speech probability per frame

    @property
    def f0(self) -> int:
        return int(round(self.start * FR))

    @property
    def end(self) -> float:
        return self.start + len(self.probs) / FR

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2


# A sampler turns [(start, length), ...] into Windows (None when a window failed). The
# optional second argument is how long each window may take before it is given up on.
Sampler = Callable[..., list["Window | None"]]


@dataclass
class Piece:
    windows: list[Window]
    rows: list[np.ndarray]  # correlation curves of those windows
    grid_delta: float       # offset chosen by the Viterbi path
    delta: float = 0.0      # refined offset
    first_cue: int = 0


@dataclass
class SyncResult:
    cues: list[Cue]
    score: float
    ratio: float
    pieces: list[dict] = field(default_factory=list)
    windows_used: int = 0

    def summary(self) -> str:
        parts = ", ".join(f"from {p['from']:.0f}s shift {-p['delta']:+.2f}s" for p in self.pieces)
        return f"score={self.score:.3f} ratio={self.ratio:.4f} windows={self.windows_used} [{parts}]"


def snap(t: float) -> float:
    return round(t * FR) / FR


# --------------------------------------------------------------------------- correlation

class Mask:
    """Subtitle speech mask on the frame grid, in ratio-corrected subtitle time."""

    def __init__(self, cues: list[Cue], ratio: float, span: float):
        self.pad = int(math.ceil((MAX_OFFSET + 2 * WINDOW_SECONDS) * FR))
        n = int(math.ceil(span * FR)) + 2 * self.pad
        m = np.zeros(n)
        for c in cues:
            if c.is_speech:
                a = self.pad + int(round(c.start / ratio * FR))
                b = self.pad + int(round(c.end / ratio * FR))
                m[max(a, 0):max(min(b, n), 0)] = 1.0
        self.m = m

    def segment(self, f_lo: int, f_hi: int) -> np.ndarray:
        """Mask frames [f_lo, f_hi) of subtitle time, zero-padded outside."""
        a, b = f_lo + self.pad, f_hi + self.pad
        out = np.zeros(f_hi - f_lo)
        ca, cb = max(a, 0), min(b, len(self.m))
        if cb > ca:
            out[ca - a:cb - a] = self.m[ca:cb]
        return out


D_MAX = int(math.ceil(MAX_OFFSET * FR))
GRID = np.arange(-D_MAX, D_MAX + 1) / FR   # candidate deltas (seconds)


def _grid_index(delta: float) -> int:
    return int(np.clip(round(delta * FR) + D_MAX, 0, len(GRID) - 1))


def correlate(win: Window, mask: Mask) -> np.ndarray:
    """Pearson correlation of window speech vs. the mask for every delta in GRID."""
    v = win.probs.astype(np.float64)
    v = v - v.mean()
    L, D = len(v), len(GRID)
    nv = float(np.sqrt((v * v).sum()))
    if nv < 1.0 or L < FR * 5:
        return np.zeros(D)
    seg = mask.segment(win.f0 - D_MAX, win.f0 + D_MAX + L)
    n = 1 << int(math.ceil(math.log2(len(seg) + L)))
    num = np.fft.irfft(np.fft.rfft(seg, n) * np.conj(np.fft.rfft(v, n)), n)[:D]
    cs = np.concatenate([[0.0], np.cumsum(seg)])
    s1 = cs[L:L + D] - cs[:D]
    var = s1 - s1 * s1 / L
    # Ignore offsets where the subtitle is (almost) all speech or all silence: no information.
    return np.where(var > MASK_VAR_GUARD * L, num / (nv * np.sqrt(np.maximum(var, 1e-9))), 0.0)


def _curves(windows: list[Window], mask: Mask, hyps: dict[float, tuple[float, ...]]) -> np.ndarray:
    R = np.stack([correlate(w, mask) for w in windows])
    # Remove each window's baseline so dense-dialogue windows don't dominate, and
    # trust windows with only a second or two of speech less: they match anything.
    R -= np.median(R, axis=1, keepdims=True)
    speech = np.array([(w.probs > 0.5).sum() / FR for w in windows])
    R *= np.clip(speech / MIN_SPEECH_SECONDS, 0.05, 1.0)[:, None]
    # Probe windows may only side with one of the pieces they were sent to separate.
    r = int(HYPOTHESIS_RADIUS * FR)
    for i, w in enumerate(windows):
        allowed = hyps.get(w.start)
        if allowed:
            keep = np.zeros(len(GRID), dtype=bool)
            for d in allowed:
                c = _grid_index(d)
                keep[max(c - r, 0):c + r + 1] = True
            R[i, ~keep] -= 5.0
    return R


# --------------------------------------------------------------------------- viterbi

def _sliding_max(x: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Max of x over [i - r, i + r] and where it is, for every i (van Herk / Gil-Werman)."""
    n, w = len(x), 2 * r + 1
    m = -(-(n + 2 * r) // w) * w
    xp = np.full(m, -np.inf)
    xp[r:r + n] = x
    pos = (np.arange(m) - r).reshape(-1, w)
    blocks = xp.reshape(-1, w)
    pre = np.maximum.accumulate(blocks, axis=1)
    pre_i = np.maximum.accumulate(np.where(blocks == pre, pos, -1), axis=1)
    rev = blocks[:, ::-1]
    suf = np.maximum.accumulate(rev, axis=1)
    suf_i = np.minimum.accumulate(np.where(rev == suf, pos[:, ::-1], m), axis=1)[:, ::-1]
    pre, pre_i, suf, suf_i = pre.ravel(), pre_i.ravel(), suf[:, ::-1].ravel(), suf_i.ravel()
    i = np.arange(n)
    take_right = pre[i + 2 * r] > suf[i]
    return (np.where(take_right, pre[i + 2 * r], suf[i]),
            np.where(take_right, pre_i[i + 2 * r], suf_i[i]))


def viterbi(R: np.ndarray) -> tuple[np.ndarray, float]:
    """One grid index per window maximising sum(R[i, path[i]]) minus movement costs.

    The first window may start at any offset."""
    n, S = R.shape
    score = R[0].copy()
    back = np.empty((n, S), dtype=np.int32)
    idx = np.arange(S)
    for i in range(1, n):
        left = np.concatenate([[-np.inf], score[:-1]]) - STEP_COST
        right = np.concatenate([score[1:], [-np.inf]]) - STEP_COST
        stack = np.stack([left, score, right])
        k = np.argmax(stack, axis=0)
        best, src = stack[k, idx], idx + k - 1
        for reach, cost in JUMPS:
            nv, ni = _sliding_max(score, int(reach * FR))
            nv = nv - cost
            use = nv > best
            best, src = np.where(use, nv, best), np.where(use, ni, src)
        j = int(np.argmax(score))
        use = score[j] - FAR_JUMP_COST > best
        best, src = np.where(use, score[j] - FAR_JUMP_COST, best), np.where(use, j, src)
        back[i] = src
        score = best + R[i]
    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path, float(score.max())


# --------------------------------------------------------------------------- pieces

def _value_near(row: np.ndarray, delta: float, radius: float = SAME_OFFSET) -> float:
    c, r = _grid_index(delta), int(radius * FR)
    return float(row[max(c - r, 0):c + r + 1].max())


def _pieces(windows: list[Window], R: np.ndarray, path: np.ndarray) -> list[Piece]:
    pieces: list[Piece] = []
    for w, row, p in zip(windows, R, path):
        d = float(GRID[p])
        if pieces and abs(d - pieces[-1].grid_delta) <= SAME_OFFSET:
            pieces[-1].windows.append(w)
            pieces[-1].rows.append(row)
        else:
            pieces.append(Piece([w], [row], d))
    return pieces


def _speech_seconds(w: Window) -> float:
    return float((w.probs > 0.5).sum() / FR)


def _jump_scale(pieces: list[Piece], i: int) -> float:
    """0.25..1: how big piece i's jump from its neighbours is (10 s or more counts as big).

    Small jumps are common in real subtitles and cheap to get wrong; big ones are what
    chance matches look like, so they must be backed by more evidence."""
    p = pieces[i]
    others = [q.grid_delta for q in pieces[max(i - 1, 0):i + 2] if q is not p]
    if not others:
        return 1.0
    return float(np.clip(min(abs(p.grid_delta - d) for d in others) / 10.0, 0.25, 1.0))


def _support(pieces: list[Piece], i: int) -> tuple[list[Window], float]:
    """Windows with real speech that clearly prefer piece i's offset over its neighbours',
    and how much they prefer it in total."""
    p = pieces[i]
    others = [q.grid_delta for q in pieces[max(i - 1, 0):i + 2] if q is not p]
    need = 0.1 * _jump_scale(pieces, i)
    out, evidence = [], 0.0
    for w, row in zip(p.windows, p.rows):
        if _speech_seconds(w) >= MIN_SPEECH_SECONDS / 2:
            own = _value_near(row, p.grid_delta)
            margin = own - max((_value_near(row, d) for d in others), default=0.0)
            if not others or margin >= need:
                out.append(w)
                evidence += margin if others else own
    return out, evidence


def _trusted(pieces: list[Piece], i: int) -> bool:
    wins, evidence = _support(pieces, i)
    return len(wins) >= 2 and evidence >= MIN_PIECE_EVIDENCE * _jump_scale(pieces, i)


def _drop_weak_pieces(pieces: list[Piece]) -> list[Piece]:
    """Fold every piece that fewer than two windows vouch for into its best neighbour.

    One window is not enough: chance matches happen (little speech, a song, a window
    straddling a jump). Real pieces collect several windows once we probe around them."""
    pieces = list(pieces)
    while len(pieces) > 1:
        weak = [(_support(pieces, i)[1], i) for i in range(len(pieces)) if not _trusted(pieces, i)]
        if not weak:
            break
        _, i = min(weak)
        p = pieces.pop(i)
        nbrs = [j for j in (i - 1, i) if 0 <= j < len(pieces)]
        j = max(nbrs, key=lambda j: sum(_value_near(r, pieces[j].grid_delta) for r in p.rows))
        q = pieces[j]
        if j >= i:
            q.windows, q.rows = p.windows + q.windows, p.rows + q.rows
        else:
            q.windows, q.rows = q.windows + p.windows, q.rows + p.rows
        # The two neighbours may now agree with each other.
        if 0 < i < len(pieces) and abs(pieces[i - 1].grid_delta - pieces[i].grid_delta) <= SAME_OFFSET:
            pieces[i - 1].windows += pieces[i].windows
            pieces[i - 1].rows += pieces[i].rows
            del pieces[i]
    return pieces


def _free(lo: float, hi: float, tried: list[tuple[float, float]]) -> tuple[float, float]:
    """Widest stretch of [lo, hi] that no decoded (or failed) window covers."""
    best, cur = (lo, lo), lo
    for s, e in sorted((s, s + ln) for s, ln in tried if s < hi and s + ln > lo):
        if s - cur > best[1] - best[0]:
            best = (cur, s)
        cur = max(cur, e)
    if hi - cur > best[1] - best[0]:
        best = (cur, hi)
    return best


def _probes(pieces: list[Piece], tried: list[tuple[float, float]],
            duration: float) -> list[tuple[float, float, tuple[float, ...]]]:
    """Windows to decode, each with the offsets it may choose between: inside every jump
    not yet pinned down, and next to lone windows (to confirm or refute them)."""
    probes = []
    for i in range(len(pieces) - 1):
        a, b = _free(pieces[i].windows[-1].end, pieces[i + 1].windows[0].start, tried)
        # A small jump needs less precision: misplacing it costs little.
        jump = abs(pieces[i].grid_delta - pieces[i + 1].grid_delta)
        resolution = GAP_RESOLVED * (4 if jump < 2 else 1)
        gap = b - a
        if gap > resolution:
            # Several evenly spaced probes per round: each round costs a full torrent
            # round-trip, so pin the jump down in as few rounds as possible.
            k = int(min(PROBES_PER_GAP, math.ceil(gap / resolution) - 1)) or 1
            spacing = gap / (k + 1)
            length = min(WINDOW_SECONDS, spacing)
            for j in range(1, k + 1):
                probes.append((snap(a + j * spacing - length / 2), length,
                               (pieces[i].grid_delta, pieces[i + 1].grid_delta)))
    if len(pieces) > 1:
        for i, p in enumerate(pieces):
            if _trusted(pieces, i):
                continue
            support = _support(pieces, i)[0]
            hyp = tuple(q.grid_delta for q in pieces[max(i - 1, 0):i + 2])
            for w in support or [w for w in p.windows if _speech_seconds(w) >= 1]:
                for a, b in ((max(w.start - WINDOW_SECONDS - 1, 0), w.start),
                             (w.end, min(w.end + WINDOW_SECONDS + 1, duration))):
                    fa, fb = _free(a, b, tried)
                    if fb - fa >= 10:
                        probes.append((snap(fa), min(WINDOW_SECONDS, fb - fa), hyp))
    seen, out = set(), []
    for s, ln, hyp in probes:
        if s not in seen:
            seen.add(s)
            out.append((s, ln, hyp))
    return out


def _refine(p: Piece) -> None:
    """Precise offset for a piece: the joint correlation peak of all its windows."""
    total = np.sum(p.rows, axis=0)
    c, r = _grid_index(p.grid_delta), int(0.5 * FR)
    lo = max(c - r, 1)
    i = lo + int(np.argmax(total[lo:min(c + r + 1, len(total) - 1)]))
    den = total[i - 1] - 2 * total[i] + total[i + 1]
    frac = float(np.clip(0.5 * (total[i - 1] - total[i + 1]) / den, -0.5, 0.5)) if den < 0 else 0.0
    p.delta = float(GRID[i] + frac / FR)


def _span_probs(windows: list[Window], f_lo: int, f_hi: int) -> np.ndarray:
    out = np.full(f_hi - f_lo, np.nan)
    for w in windows:
        a = w.f0 - f_lo
        s0, s1 = max(a, 0), min(a + len(w.probs), len(out))
        if s1 > s0:
            out[s0:s1] = w.probs[s0 - a:s1 - a]
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~np.isnan(x)
    if ok.sum() < FR * 5:
        return 0.0
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    den = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / den) if den > 1e-9 else 0.0


def _choose_split(cues: list[Cue], ratio: float, A: Piece, B: Piece, windows: list[Window]) -> int:
    """Index of the first cue that belongs to piece B, judged against the audio around the jump."""
    times = np.array([[c.start / ratio, c.end / ratio] for c in cues])
    speech = np.array([c.is_speech for c in cues])
    lw, rw = A.windows[-1], B.windows[0]
    u_lo, u_hi = sorted((lw.start + A.delta, rw.end + B.delta))
    cand = [j for j in range(len(cues)) if u_lo - 5 <= times[j, 0] <= u_hi + 5]
    if not cand:
        return int(np.searchsorted(times[:, 0], (u_lo + u_hi) / 2))
    cand.append(cand[-1] + 1)
    f_lo, f_hi = lw.f0, rw.f0 + len(rw.probs)
    probs = _span_probs([w for w in windows if w.end > lw.start and w.start < rw.end], f_lo, f_hi)
    near = [k for k in range(max(cand[0] - 40, 0), min(cand[-1] + 40, len(cues))) if speech[k]]
    best, best_j = -2.0, cand[0]
    for j in cand:
        pred = np.zeros(f_hi - f_lo)
        for k in near:
            d = A.delta if k < j else B.delta
            a = int(round((times[k, 0] - d) * FR)) - f_lo
            b = int(round((times[k, 1] - d) * FR)) - f_lo
            if b > 0 and a < len(pred):
                pred[max(a, 0):min(b, len(pred))] = 1
        score = _pearson(probs, pred)
        if score > best + 1e-6:
            best, best_j = score, j
    return best_j


# --------------------------------------------------------------------------- driver

_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="align")


def initial_plan(duration: float, max_windows: int = MAX_WINDOWS) -> list[tuple[float, float]]:
    n = int(np.clip(round(duration / SAMPLE_SPACING), MIN_WINDOWS, max(MIN_WINDOWS, max_windows)))
    lo, hi = duration * 0.01, duration * 0.97 - WINDOW_SECONDS
    step = (hi - lo) / n
    return [(snap(lo + (i + INITIAL_PHASE) * step - WINDOW_SECONDS / 2), WINDOW_SECONDS) for i in range(n)]


def _usable(ws) -> list[Window]:
    return [w for w in ws if w is not None and len(w.probs) > FR * 5]


def sync(cues: list[Cue], sample: Sampler, duration: float, max_windows: int = MAX_WINDOWS) -> SyncResult:
    """Re-time `cues` to the video. `max_windows` caps the initial windows (and, likewise,
    the extra ones) to limit how much of a big file gets downloaded."""
    if not cues:
        raise ValueError("subtitle has no cues")
    span = max(duration, max(c.end for c in cues)) + 60

    windows = sorted(_usable(sample(initial_plan(duration, max_windows))), key=lambda w: w.start)
    max_extra = min(MAX_EXTRA_WINDOWS, max(max_windows, MIN_WINDOWS))
    if not windows:
        raise RuntimeError("could not read any audio from the video")
    tried = [(w.start, len(w.probs) / FR) for w in windows]
    hyps: dict[float, tuple[float, ...]] = {}

    # 1. Framerate ratio: whichever explains the audio best along its Viterbi path.
    def evaluate(ratio: float):
        mask = Mask(cues, ratio, span)
        R = _curves(windows, mask, hyps)
        path, score = viterbi(R)
        return score, ratio, mask, R, path

    def best_of(ratios, bias):
        results = list(_POOL.map(evaluate, ratios))
        return max(results, key=lambda r: r[0] * (bias if r[1] == ratios[0] else 1.0))

    # Standard framerate conversions (preferring none), then slow drift around the winner:
    # coarse steps first, then finer ones around the best.
    best = best_of([1.0] + [r for r in RATIOS if r != 1.0], 1.1)
    step = FINE_RATIO / 4
    for _ in range(3):
        base = best[1]
        best = best_of([base] + [base * (1 + k * step) for k in (-4, -3, -2, -1, 1, 2, 3, 4)
                                 if abs(base * (1 + k * step) / best[1] - 1) <= FINE_RATIO], 1.02)
        step /= 4
    _, ratio, mask, R, path = best
    log.debug("ratio %.5f from %d windows", ratio, len(windows))

    # 2. Pieces, bisecting every jump until it is pinned down.
    pieces = _pieces(windows, R, path)
    # A subtitle that matches nothing (wrong movie, broken file) isn't worth more audio.
    rough = _drop_weak_pieces([Piece(list(p.windows), list(p.rows), p.grid_delta) for p in pieces])
    for p in rough:
        p.delta = p.grid_delta
    hopeless = quality(apply(cues, ratio, rough), windows) < HOPELESS_SCORE
    extra = max_extra if hopeless else 0
    while extra < max_extra:
        probes = _probes(pieces, tried, duration)[: max_extra - extra]
        if not probes:
            break
        extra += len(probes)
        tried += [(s, ln) for s, ln, _ in probes]
        got = sample([(s, ln) for s, ln, _ in probes], PROBE_TIMEOUT)
        new = []
        for (s, ln, hyp), w in zip(probes, got):
            if w is not None and len(w.probs) > FR * 5:
                hyps[w.start] = hyp
                new.append(w)
        if new:
            windows = sorted(windows + new, key=lambda w: w.start)
            R = _curves(windows, mask, hyps)
            path, _ = viterbi(R)
            pieces = _pieces(windows, R, path)

    pieces = _drop_weak_pieces(pieces)
    for p in pieces:
        _refine(p)

    # 3. Exact cue at each jump.
    for i in range(1, len(pieces)):
        pieces[i].first_cue = max(pieces[i - 1].first_cue,
                                  _choose_split(cues, ratio, pieces[i - 1], pieces[i], windows))

    out = apply(cues, ratio, pieces)
    return SyncResult(
        cues=out, score=quality(out, windows), ratio=ratio, windows_used=len(windows),
        pieces=[{"from": p.windows[0].start, "cue": p.first_cue, "delta": p.delta,
                 "windows": len(p.windows)} for p in pieces],
    )


def apply(cues: list[Cue], ratio: float, pieces: list[Piece]) -> list[Cue]:
    out = []
    k = 0
    for i, c in enumerate(cues):
        while k + 1 < len(pieces) and i >= pieces[k + 1].first_cue:
            k += 1
        d = pieces[k].delta
        out.append(Cue(c.start / ratio - d, c.end / ratio - d, c.text))
    return out


def quality(cues: list[Cue], windows: list[Window]) -> float:
    """Correlation between the synced subtitle and the sampled speech it covers (higher is better)."""
    spans = np.array([[c.start, c.end] for c in cues if c.is_speech])
    if not windows or not len(spans):
        return 0.0
    lo, hi = spans[:, 0].min() - 30, spans[:, 1].max() + 30
    windows = [w for w in windows if w.start >= lo and w.end <= hi] or windows
    order = np.argsort(spans[:, 0])
    starts, ends = spans[order, 0], np.maximum.accumulate(spans[order, 1])
    xs, ys = [], []
    for w in windows:
        t = w.start + np.arange(len(w.probs)) / FR
        i = np.searchsorted(starts, t, side="right") - 1
        ys.append(((i >= 0) & (t < ends[np.maximum(i, 0)])).astype(np.float64))
        xs.append(w.probs.astype(np.float64))
    return _pearson(np.concatenate(xs), np.concatenate(ys))

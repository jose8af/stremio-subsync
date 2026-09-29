"""Sync a subtitle file against a video file or URL from the command line.

    python -m subsync.cli VIDEO SUBTITLE [-o OUT.srt]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from . import align, subtitles
from .sampler import VideoAudio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("subtitle")
    ap.add_argument("-o", "--output")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    t0 = time.monotonic()
    cues = subtitles.parse(subtitles.decode(Path(args.subtitle).read_bytes()))
    video = VideoAudio(args.video)
    res = align.sync(cues, video.sample, video.duration)
    print(f"{res.summary()}  audio={video.seconds_fetched:.0f}s of {video.duration:.0f}s  "
          f"time={time.monotonic() - t0:.1f}s")
    out = args.output or str(Path(args.subtitle).with_suffix(".synced.srt"))
    Path(out).write_text(subtitles.to_srt(res.cues), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()

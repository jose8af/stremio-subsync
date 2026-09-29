# SubSync: auto-synced subtitles for Stremio (macOS, local)

A local Stremio addon. When you play a torrent, it downloads subtitles from
OpenSubtitles (the same source Stremio uses), listens to short pieces of the video's
audio, and re-times each subtitle to match. Synced subtitles appear in Stremio's
subtitle menu under **SubSync**.

It fixes the common kinds of bad sync:

| Problem | Example | Handled by |
|---|---|---|
| Constant delay | subs 30 s late | piece offset |
| Framerate mismatch (the delay grows through the movie) | subs made for 25 fps, video 23.976 | ratio detection (13 standard ratios) |
| Odd slow drift | 19 s off by the end | fine ratio search (±0.4 %) |
| Delay that changes mid-video | different cut, intro, ad breaks, missing scene | piecewise offsets with the exact cue where it changes |

## Why it's fast

SubSync reads only 30-second windows spread through the video: 10 to 40 windows depending
on length, plus a few extra around any jump it finds. That's roughly 15–20 % of the file,
fetched 10 windows at a time. The count is capped (40, plus at most 40 extra), so a 3-hour
movie costs about the same as a 2-hour one. ffmpeg seeks with HTTP range requests, so only
those parts of the torrent are downloaded, and Stremio keeps them for playback anyway.

In each window, a neural speech detector (Silero VAD) marks where people talk. All windows
are then decoded jointly (Viterbi): each window chooses an offset, and changing offset
between windows costs a penalty. So one ambiguous window can't derail the result, but a
real jump that several windows agree on is kept. Where the offset jumps, it samples in
between until it can pick the exact cue at which the jump happens.

Once the audio is in, syncing one subtitle takes about 2–4 s of CPU. Extra subtitles for
the same video reuse the audio already fetched.

Tested on 108 real OpenSubtitles files (27 languages) against two different releases of
the same movie, checked independently against full-movie speech detection. It was also
tested on randomized synthetic distortions. The typical remaining error is under 0.2 s.

## Setup

Requirements:

- macOS (the launcher, autostart and notifications are macOS-specific; the core is plain Python)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)
- [Stremio](https://www.stremio.com/) desktop, running (its local streaming server is used to read the video)

```sh
git clone https://github.com/jose8af/stremio-subsync.git
cd stremio-subsync
./run.sh    # first run creates .venv, installs numpy + onnxruntime, downloads the Silero VAD model
```

Then in Stremio: **Addons → search box → paste** `http://127.0.0.1:7979/manifest.json` → Install.
Or open http://127.0.0.1:7979 and click **Install in Stremio**.

> `127.0.0.1` is your own computer: SubSync runs locally next to Stremio, so this address is
> the same for everyone. If port 7979 is taken, pick another with `SUBSYNC_PORT=8080 ./run.sh`
> and use that port in the URLs above.

To start it automatically at login:

```sh
./install_autostart.sh            # SUBSYNC_LANGUAGES=eng,spa ./install_autostart.sh to change languages
./install_autostart.sh --remove
```

Logs are written to `~/Library/Logs/subsync.log`.

## Using it

1. Play something from a torrent source (e.g. Torrentio without debrid). SubSync starts
   working as soon as the player opens.
2. When the first subtitle of each language is ready, you get a macOS notification
   ("Spanish & English ready after 80s: pick SubSync #1"). The page at
   http://127.0.0.1:7979 also shows **Ready to watch** per language.
3. In the subtitle menu, pick a language, then **SubSync #1 (best)**. #2 and #3 are other
   subtitle files, ranked by score, in case #1 doesn't suit you.
4. If you select one before it's ready, Stremio waits (up to 90 s) and then shows it.
   If you see "still syncing", select it again a bit later.

**Order:** #1, #2, #3 are sorted by score. The one exception: if you pick #1 before all
subtitles of that language have finished, #1 stays the subtitle you got, even if a
better-scoring one finishes later (that one becomes #2). The status page always lists
them in the same order as Stremio.

**Score:** how well the synced subtitle's timing matches the speech in the video. It is
the correlation between "someone is speaking" (from the speech detector) and "a subtitle
line is on screen", over all the audio samples SubSync read. 1.0 would be a perfect
match and 0 means no relation. Good syncs usually score 0.5–0.75, because subtitles stay
on screen during short pauses, and some speech is never subtitled (background voices,
singing). Below 0.3 a subtitle is rejected. Use the score to compare subtitles for the
same video, not across videos: a talky episode scores higher than a noisy action movie.

Results are cached in `~/Library/Caches/stremio-subsync`, so reopening a video is instant.

**Languages:** set them in the install URL, e.g. `http://127.0.0.1:7979/eng,spa/manifest.json`.
Use OpenSubtitles' 3-letter codes: `eng spa pob por fre ger ita dut pol rus tur ara heb ...`.

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `SUBSYNC_LANGUAGES` | `spa,eng` | default languages when the URL has none |
| `SUBSYNC_CANDIDATES` | `3` | subtitle files synced per language |
| `SUBSYNC_PORT` | `7979` | |
| `SUBSYNC_WORKERS` | `10` | parallel audio readers |
| `SUBSYNC_PARALLEL_SYNCS` | `4` | subtitles synced at the same time |
| `SUBSYNC_EARLY_SCORE` | `0.5` | hand out subtitle #1 as soon as one scores this well |
| `SUBSYNC_NOTIFY` | `1` | macOS notification when #1 is ready (`0` to turn off) |
| `SUBSYNC_BUDGET_MB` | `700` | rough cap on how much of the video the samples download (very high-bitrate files get fewer samples) |
| `SUBSYNC_WAIT` | `90` | seconds a subtitle request waits for a sync in progress |
| `SUBSYNC_MIN_SCORE` | `0.3` | results scoring below this are treated as failed (good ones score 0.5–0.75) |
| `SUBSYNC_STREMIO_SERVER` | `http://127.0.0.1:11470` | Stremio's streaming server |

## Command line

Sync a local file, with no Stremio needed:

```sh
.venv/bin/python -m subsync.cli movie.mkv movie.srt -o movie.synced.srt
```

## Limits

- **Torrents only (for now).** The addon finds the playing file through Stremio's local
  streaming server (`127.0.0.1:11470`). Debrid/HTTP streams don't pass through it, so the
  addon can't see them and simply lists nothing.
- A delay change that lasts less than about 5 minutes (one or two windows) can be missed.
- Subtitles for a different movie or a broken file are detected (low score) and not offered.

## Layout

```
subsync/align.py      the sync algorithm
subsync/vad.py        Silero speech detection (ONNX)
subsync/media.py      ffprobe / ffmpeg window extraction
subsync/sampler.py    cached parallel audio windows for one video
subsync/sources.py    OpenSubtitles candidates and ranking
subsync/stremio.py    finds the playing torrent on Stremio's streaming server
subsync/jobs.py       background job per video, disk cache
subsync/server.py     the Stremio addon (HTTP)
subsync/cli.py        command-line sync
```

## Contributing

Issues and pull requests are welcome. Ideas: Linux/Windows launchers, debrid/HTTP stream
support (would need a caching proxy, since remote seeks are slow), more subtitle sources.

## Credits and license

- Speech detection: [Silero VAD](https://github.com/snakers4/silero-vad) (MIT), downloaded on first run.
- Subtitles come from the OpenSubtitles addon that Stremio already uses. SubSync is not
  affiliated with Stremio or OpenSubtitles.

Released under the [MIT License](LICENSE).

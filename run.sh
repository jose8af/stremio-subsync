#!/bin/zsh
# Start the SubSync Stremio addon (creates the virtualenv on first run).
set -e
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null && [[ ! -x /opt/homebrew/bin/ffmpeg ]]; then
  echo "ffmpeg is required: brew install ffmpeg" >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
if [[ ! -f models/silero_vad.onnx ]]; then
  mkdir -p models
  curl -fsSL -o models/silero_vad.onnx \
    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
fi
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec .venv/bin/python -m subsync.server

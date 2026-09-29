"""Speech detection with the Silero VAD ONNX model (robust to music and effects)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
CHUNK = 512  # samples per model step at 16 kHz
CONTEXT = 64
FRAME_RATE = SAMPLE_RATE / CHUNK  # 31.25 speech-probability frames per second

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "silero_vad.onnx"


class SileroVAD:
    def __init__(self, model_path: Path = MODEL_PATH):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                          providers=["CPUExecutionProvider"])
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._lock = threading.Lock()

    def speech_probs(self, audio: np.ndarray) -> np.ndarray:
        """Return one speech probability per 32 ms frame of 16 kHz mono float32 audio."""
        n = len(audio) // CHUNK
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        frames = audio[: n * CHUNK].astype(np.float32).reshape(n, CHUNK)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT), dtype=np.float32)
        out = np.empty(n, dtype=np.float32)
        run = self._sess.run
        for i in range(n):
            x = np.concatenate([context, frames[i:i + 1]], axis=1)
            prob, state = run(None, {"input": x, "state": state, "sr": self._sr})
            out[i] = prob[0, 0]
            context = x[:, -CONTEXT:]
        return out


_instance: SileroVAD | None = None
_instance_lock = threading.Lock()


def get_vad() -> SileroVAD:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = SileroVAD()
        return _instance

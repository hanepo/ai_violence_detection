from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple

import librosa
import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover
    sd = None


class AudioStream:
    def __init__(self, sample_rate: int, channels: int, chunk_seconds: float):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_seconds = chunk_seconds
        self._input_device = None
        self._input_sample_rate = None

    def _capture(self, n_samples: int, samplerate: int, device=None) -> np.ndarray:
        chunk = sd.rec(
            n_samples,
            samplerate=samplerate,
            channels=self.channels,
            dtype="float32",
            device=device,
        )
        sd.wait()
        return chunk.squeeze()

    def read_chunk(self) -> tuple[np.ndarray, datetime]:
        if sd is None:
            n = int(self.sample_rate * self.chunk_seconds)
            return np.zeros((n,), dtype=np.float32), datetime.now(timezone.utc)

        n = int(self.sample_rate * self.chunk_seconds)
        try:
            data = self._capture(n, self.sample_rate)
            return data.astype(np.float32), datetime.now(timezone.utc)
        except Exception:
            # Some USB microphones reject low sample rates (e.g., 16 kHz).
            # Capture at the input device native rate and resample to model rate.
            try:
                if self._input_device is None:
                    default_input = sd.default.device[0] if sd.default.device else None
                    self._input_device = default_input if default_input is not None and default_input >= 0 else None

                if self._input_sample_rate is None:
                    info = sd.query_devices(self._input_device, "input")
                    self._input_sample_rate = int(info.get("default_samplerate", self.sample_rate))

                native_sr = max(int(self._input_sample_rate), 8000)
                native_n = int(native_sr * self.chunk_seconds)
                native_data = self._capture(native_n, native_sr, self._input_device)

                if native_data.ndim > 1:
                    native_data = native_data.mean(axis=1)

                if native_sr != self.sample_rate:
                    data = librosa.resample(native_data.astype(np.float32), orig_sr=native_sr, target_sr=self.sample_rate)
                else:
                    data = native_data.astype(np.float32)

                if data.shape[0] < n:
                    data = np.pad(data, (0, n - data.shape[0]))
                elif data.shape[0] > n:
                    data = data[:n]

                return data.astype(np.float32), datetime.now(timezone.utc)
            except Exception:
                return np.zeros((n,), dtype=np.float32), datetime.now(timezone.utc)


def extract_mfcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 40) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)

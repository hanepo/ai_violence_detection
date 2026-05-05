from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np


class LiteModelRunner:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.backend = None

        use_tflite = os.getenv("ENABLE_TFLITE", "1") == "1"
        if use_tflite and Path(model_path).exists():
            try:
                # Prefer lightweight runtime on ARM/Raspberry Pi.
                from tflite_runtime.interpreter import Interpreter

                self.interpreter = Interpreter(model_path=model_path)
                self.backend = "tflite_runtime"
            except Exception:
                try:
                    # Python 3.13 on Pi OS can use LiteRT instead of tflite-runtime.
                    from ai_edge_litert.interpreter import Interpreter

                    self.interpreter = Interpreter(model_path=model_path)
                    self.backend = "ai_edge_litert"
                except Exception:
                    try:
                        import tensorflow as tf

                        self.interpreter = tf.lite.Interpreter(model_path=model_path)
                        self.backend = "tensorflow"
                    except Exception:
                        self.interpreter = None
                        self.backend = None

            if self.interpreter is not None:
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()

    def available(self) -> bool:
        return self.interpreter is not None

    def predict_score(self, x: np.ndarray) -> float:
        y = self.predict_vector(x)
        if y.size == 0:
            return 0.0
        if y.size == 1:
            return float(np.clip(y[0], 0.0, 1.0))
        return float(np.clip(y.max(), 0.0, 1.0))

    def predict_vector(self, x: np.ndarray) -> np.ndarray:
        if self.interpreter is None:
            return np.array([self._fallback_score(x)], dtype=np.float32)

        inp = self.input_details[0]
        out = self.output_details[0]
        x = x.astype(inp["dtype"])
        x = np.expand_dims(x, axis=0)

        self.interpreter.set_tensor(inp["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(out["index"])
        return np.array(y, dtype=np.float32).reshape(-1)

    @staticmethod
    def _fallback_score(x: np.ndarray) -> float:
        # Deterministic pseudo-score for dry-runs before trained models exist.
        digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()[:4096]).digest()
        value = int.from_bytes(digest[:2], "big") / 65535.0
        return float(np.clip(value, 0.0, 1.0))


class VisionInference:
    def __init__(self, model_path: str):
        self.runner = LiteModelRunner(model_path)

    def score(self, frame: np.ndarray) -> float:
        resized = cv2.resize(frame, (224, 224))
        # The vision model includes a Rescaling(1/255) layer internally.
        # Passing raw float pixel values avoids double-normalization.
        model_input = resized.astype(np.float32)
        return self.runner.predict_score(model_input)


class AudioInference:
    def __init__(self, model_path: str):
        self.runner = LiteModelRunner(model_path)
        self.labels = self._load_labels(model_path)
        self.aggressive_index = self._resolve_aggressive_index(self.labels)

    @staticmethod
    def _load_labels(model_path: str) -> list[str]:
        label_path = Path(model_path).with_suffix(".labels.txt")
        if not label_path.exists():
            return []
        return [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _resolve_aggressive_index(labels: list[str]) -> int | None:
        if not labels:
            return None

        for i, label in enumerate(labels):
            norm = label.lower().replace("-", "_").replace(" ", "_")
            if norm == "aggressive":
                return i

        for i, label in enumerate(labels):
            norm = label.lower().replace("-", "_").replace(" ", "_")
            if "aggress" in norm or "fight" in norm or "viol" in norm:
                return i

        return None

    @staticmethod
    def _prepare_mfcc(mfcc: np.ndarray) -> np.ndarray:
        mfcc = mfcc[:40, :80]
        if mfcc.shape[1] < 80:
            pad = np.zeros((mfcc.shape[0], 80 - mfcc.shape[1]), dtype=np.float32)
            mfcc = np.concatenate([mfcc, pad], axis=1)

        mfcc = mfcc.astype(np.float32)
        mu = float(mfcc.mean())
        sigma = float(mfcc.std())
        mfcc = (mfcc - mu) / max(sigma, 1e-6)

        # CNN expects (40, 80, 1) before batch dimension.
        return np.expand_dims(mfcc, axis=-1)

    def predict(self, mfcc: np.ndarray) -> dict:
        model_input = self._prepare_mfcc(mfcc)
        y = self.runner.predict_vector(model_input)
        if y.size == 0:
            return {
                "aggressive_score": 0.0,
                "top_label": None,
                "top_score": 0.0,
                "probabilities": {},
            }

        if y.size == 1:
            score = float(np.clip(y[0], 0.0, 1.0))
            return {
                "aggressive_score": score,
                "top_label": "aggressive" if score >= 0.5 else "non_aggressive",
                "top_score": score if score >= 0.5 else 1.0 - score,
                "probabilities": {
                    "non_aggressive": float(1.0 - score),
                    "aggressive": score,
                },
            }

        probs = np.array(y, dtype=np.float32)
        total = float(probs.sum())
        if total > 0:
            probs = probs / total
        probs = np.clip(probs, 0.0, 1.0)

        top_idx = int(np.argmax(probs))
        if self.labels and top_idx < len(self.labels):
            top_label = self.labels[top_idx]
        else:
            top_label = f"class_{top_idx}"

        if self.aggressive_index is not None and self.aggressive_index < probs.size:
            aggressive_score = float(probs[self.aggressive_index])
        else:
            aggressive_score = float(probs[top_idx])

        prob_map = {}
        if self.labels and len(self.labels) == probs.size:
            for i, label in enumerate(self.labels):
                prob_map[label] = float(probs[i])

        return {
            "aggressive_score": float(np.clip(aggressive_score, 0.0, 1.0)),
            "top_label": top_label,
            "top_score": float(probs[top_idx]),
            "probabilities": prob_map,
        }

    def score(self, mfcc: np.ndarray) -> float:
        return float(self.predict(mfcc)["aggressive_score"])

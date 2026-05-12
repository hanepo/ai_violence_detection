"""Build dataset/audio from local CREMA-style WAVs (e.g. dashboard output/uploads).

Use when Kaggle is unavailable: copies labeled files and pads rare classes with
light noise so train/val/test splits stay viable.

Usage (from project root):
    python scripts/bootstrap_audio_from_uploads.py
    python scripts/bootstrap_audio_from_uploads.py --upload-dir output/uploads --min-per-class 24
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def map_cremad_label(code: str) -> str | None:
    if code == "ANG":
        return "aggressive"
    if code in {"DIS", "FEA", "SAD"}:
        return "distressed"
    if code in {"NEU", "HAP"}:
        return "neutral"
    return None


def emotion_from_stem(stem: str) -> str | None:
    for tok in stem.split("_"):
        u = tok.upper()
        if u in {"ANG", "DIS", "FEA", "HAP", "NEU", "SAD"}:
            return map_cremad_label(u)
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_wav_mono(path: Path, y: np.ndarray, sr: int) -> None:
    import wave

    y16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(y16.tobytes())


def pad_class_with_noise(class_dir: Path, target: int, rng: np.random.Generator) -> int:
    files = list(class_dir.glob("*.wav"))
    if not files:
        return 0
    added = 0
    idx = 0
    while len(list(class_dir.glob("*.wav"))) < target:
        src = files[rng.integers(0, len(files))]
        y, sr = librosa.load(str(src), sr=16000, mono=True)
        noise = rng.normal(0, 0.002, size=y.shape).astype(np.float32)
        y_aug = np.clip(y.astype(np.float32) + noise, -1.0, 1.0)
        dst = class_dir / f"_aug_{idx}_{src.stem}.wav"
        idx += 1
        if dst.exists():
            continue
        _write_wav_mono(dst, y_aug, sr)
        added += 1
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap dataset/audio from upload WAVs")
    parser.add_argument("--upload-dir", type=Path, default=ROOT / "output" / "uploads")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dataset" / "audio")
    parser.add_argument("--min-per-class", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.upload_dir.is_dir():
        print(f"Upload dir missing: {args.upload_dir}")
        return 1

    for cls in ("aggressive", "distressed", "neutral", "physical_impacts"):
        ensure_dir(args.out_dir / cls)

    copied = 0
    skipped = 0
    for wav in sorted(args.upload_dir.glob("*.wav")):
        label = emotion_from_stem(wav.stem)
        if label is None:
            skipped += 1
            continue
        dst = args.out_dir / label / wav.name
        if not dst.exists():
            shutil.copy2(wav, dst)
            copied += 1

    rng = np.random.default_rng(args.seed)
    total_aug = 0
    if args.min_per_class > 0:
        for cls in ("aggressive", "distressed", "neutral", "physical_impacts"):
            total_aug += pad_class_with_noise(args.out_dir / cls, args.min_per_class, rng)

    print(f"Bootstrap done: copied={copied} skipped_unlabeled={skipped} noise_aug_writes={total_aug}")
    print(f"Output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

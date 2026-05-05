from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_first_existing(root: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        p = root / rel
        if p.exists() and p.is_dir():
            return p
    return None


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    frame_every: int,
    max_frames_per_video: int,
    resize: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    saved = 0
    frame_idx = 0
    stem = video_path.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_every == 0:
            if resize > 0:
                frame = cv2.resize(frame, (resize, resize))
            out_file = out_dir / f"{stem}_{saved:04d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved += 1
            if max_frames_per_video > 0 and saved >= max_frames_per_video:
                break

        frame_idx += 1

    cap.release()
    return saved


def prepare_vision(
    raw_root: Path,
    out_root: Path,
    frame_every: int,
    max_frames_per_video: int,
    resize: int,
) -> None:
    violent_dir = out_root / "violent"
    non_violent_dir = out_root / "non_violent"
    ensure_dir(violent_dir)
    ensure_dir(non_violent_dir)

    dataset_root = find_first_existing(
        raw_root,
        [
            "Real Life Violence Dataset",
            "real life violence situations/Real Life Violence Dataset",
        ],
    )
    if dataset_root is None:
        raise FileNotFoundError(f"Could not locate RWF dataset under: {raw_root}")

    class_dirs = [
        ("Violence", violent_dir),
        ("NonViolence", non_violent_dir),
    ]

    total_saved = 0
    total_videos = 0
    for src_name, dst_dir in class_dirs:
        src_dir = dataset_root / src_name
        if not src_dir.exists():
            continue
        videos = [p for p in src_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
        for vp in videos:
            saved = extract_frames_from_video(
                video_path=vp,
                out_dir=dst_dir,
                frame_every=frame_every,
                max_frames_per_video=max_frames_per_video,
                resize=resize,
            )
            total_saved += saved
            total_videos += 1

    print(f"Vision prep complete. Videos processed: {total_videos}, frames saved: {total_saved}")
    print(f"Output: {out_root}")


def prepare_hockey_vision(
    raw_root: Path,
    out_root: Path,
    frame_every: int,
    max_frames_per_video: int,
    resize: int,
) -> None:
    violent_dir = out_root / "violent"
    non_violent_dir = out_root / "non_violent"
    ensure_dir(violent_dir)
    ensure_dir(non_violent_dir)

    data_dir = raw_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected hockey data dir at: {data_dir}")

    total_saved = 0
    total_videos = 0
    for vp in [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]:
        name = vp.name.lower()
        if name.startswith("fi"):
            dst = violent_dir
        elif name.startswith("no"):
            dst = non_violent_dir
        else:
            continue

        saved = extract_frames_from_video(
            video_path=vp,
            out_dir=dst,
            frame_every=frame_every,
            max_frames_per_video=max_frames_per_video,
            resize=resize,
        )
        total_saved += saved
        total_videos += 1

    print(f"Hockey prep complete. Videos processed: {total_videos}, frames saved: {total_saved}")
    print(f"Merged output: {out_root}")


def map_cremad_label(code: str) -> str | None:
    # CREMA-D emotion codes in filename: ANG, DIS, FEA, HAP, NEU, SAD
    if code == "ANG":
        return "aggressive"
    if code in {"DIS", "FEA", "SAD"}:
        return "distressed"
    if code in {"NEU", "HAP"}:
        return "neutral"
    return None


def prepare_audio(raw_root: Path, out_root: Path) -> None:
    for cls in ["aggressive", "distressed", "neutral"]:
        ensure_dir(out_root / cls)

    wavs = list(raw_root.rglob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files found under: {raw_root}")

    copied = 0
    skipped = 0
    for wav_path in wavs:
        parts = wav_path.stem.split("_")
        if len(parts) < 3:
            skipped += 1
            continue
        emotion_code = parts[2].upper()
        label = map_cremad_label(emotion_code)
        if label is None:
            skipped += 1
            continue

        dst = out_root / label / wav_path.name
        if not dst.exists():
            shutil.copy2(wav_path, dst)
            copied += 1

    print(f"Audio prep complete. Files copied: {copied}, skipped: {skipped}")
    print(f"Output: {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw datasets into training-ready folders")
    parser.add_argument("--vision", action="store_true", help="Prepare vision dataset")
    parser.add_argument("--audio", action="store_true", help="Prepare audio dataset")
    parser.add_argument("--all", action="store_true", help="Prepare both")
    parser.add_argument("--extra-hockey", action="store_true", help="Prepare extra hockey fight dataset and merge into vision")

    parser.add_argument("--rwf-raw", default="dataset/raw/rwf2000")
    parser.add_argument("--hockey-raw", default="dataset/raw/extra_hockey")
    parser.add_argument("--cremad-raw", default="dataset/raw/cremad")
    parser.add_argument("--vision-out", default="dataset/vision")
    parser.add_argument("--audio-out", default="dataset/audio")

    parser.add_argument("--frame-every", type=int, default=45)
    parser.add_argument("--max-frames-per-video", type=int, default=10)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any([args.vision, args.audio, args.all, args.extra_hockey]):
        print("Select --vision, --audio, --extra-hockey, or --all")
        return 1

    if args.all or args.vision:
        prepare_vision(
            raw_root=Path(args.rwf_raw),
            out_root=Path(args.vision_out),
            frame_every=max(1, args.frame_every),
            max_frames_per_video=max(1, args.max_frames_per_video),
            resize=max(1, args.resize),
        )

    if args.extra_hockey:
        prepare_hockey_vision(
            raw_root=Path(args.hockey_raw),
            out_root=Path(args.vision_out),
            frame_every=max(1, args.frame_every),
            max_frames_per_video=max(1, args.max_frames_per_video),
            resize=max(1, args.resize),
        )

    if args.all or args.audio:
        prepare_audio(
            raw_root=Path(args.cremad_raw),
            out_root=Path(args.audio_out),
        )

    print("Dataset preparation done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

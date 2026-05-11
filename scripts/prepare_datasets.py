from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}


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


def prepare_cctv_fights_vision(
    cctv_root: Path,
    nonfight_dir: Path,
    out_root: Path,
    violent_frame_every: int = 15,
    violent_max_frames: int = 10,
    nonfight_frame_every: int = 10,
    nonfight_max_frames: int = 50,
    resize: int = 224,
) -> None:
    """Extract frames from CCTV Fights Dataset (violent) and a non-fight video dir.

    CCTV Fights Dataset layout expected under cctv_root:
        CCTV_DATA/{training,testing,validation}/*.mpeg
        NON_CCTV_DATA/{training,testing,validation}/*.mpeg
    All videos are treated as the 'violent' class.
    Videos in nonfight_dir are the 'non_violent' class.
    """
    violent_dir = out_root / "violent"
    non_violent_dir = out_root / "non_violent"
    ensure_dir(violent_dir)
    ensure_dir(non_violent_dir)

    fight_videos = [
        p for p in cctv_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    print(f"Found {len(fight_videos)} fight videos in {cctv_root}")
    total_violent = 0
    for vp in sorted(fight_videos):
        total_violent += extract_frames_from_video(
            vp, violent_dir, violent_frame_every, violent_max_frames, resize
        )

    nonfight_videos = [
        p for p in nonfight_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    print(f"Found {len(nonfight_videos)} non-fight videos in {nonfight_dir}")
    total_nonviolent = 0
    for vp in sorted(nonfight_videos):
        total_nonviolent += extract_frames_from_video(
            vp, non_violent_dir, nonfight_frame_every, nonfight_max_frames, resize
        )

    print(f"CCTV vision prep complete.")
    print(f"  Violent frames   : {total_violent}")
    print(f"  Non-violent frames: {total_nonviolent}")
    print(f"  Output: {out_root}")


def merge_youtube_violence_wavs(raw_root: Path, audio_out: Path, max_files: int = 4000) -> int:
    """Append WAVs from Kaggle 'Audio-based Violence Detection' Violence/ folders into aggressive/."""
    aggressive_dir = audio_out / "aggressive"
    ensure_dir(aggressive_dir)
    violence_dirs = [p for p in raw_root.rglob("*") if p.is_dir() and p.name.lower() == "violence"]
    n = 0
    for vdir in violence_dirs:
        for wav_path in vdir.glob("*.wav"):
            if n >= max_files:
                return n
            dst = aggressive_dir / f"ytviol_{n:06d}_{wav_path.name}"
            if not dst.exists():
                shutil.copy2(wav_path, dst)
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw datasets into training-ready folders")
    parser.add_argument("--vision", action="store_true", help="Prepare vision dataset")
    parser.add_argument("--audio", action="store_true", help="Prepare audio dataset")
    parser.add_argument("--all", action="store_true", help="Prepare both")
    parser.add_argument("--extra-hockey", action="store_true", help="Prepare extra hockey fight dataset and merge into vision")
    parser.add_argument("--cctv-fights", action="store_true", help="Prepare CCTV Fights Dataset (violent) + Peliculas noFights (non_violent)")

    parser.add_argument("--cctv-root", default="dataset/vision dataset", help="Root of CCTV Fights Dataset (has CCTV_DATA/ and NON_CCTV_DATA/)")
    parser.add_argument("--nonfight-root", default="dataset/Peliculas/noFights", help="Directory of non-fight videos")
    parser.add_argument("--rwf-raw", default="dataset/raw/rwf2000")
    parser.add_argument("--hockey-raw", default="dataset/raw/extra_hockey")
    parser.add_argument("--cremad-raw", default="dataset/raw/cremad")
    parser.add_argument("--violence-raw", default="dataset/raw/audio_violence_youtube")
    parser.add_argument("--vision-out", default="dataset/vision")
    parser.add_argument("--audio-out", default="dataset/audio")

    parser.add_argument(
        "--merge-youtube-violence",
        action="store_true",
        help="After --audio, also copy Violence/*.wav from --violence-raw into audio/aggressive/",
    )

    parser.add_argument("--frame-every", type=int, default=45)
    parser.add_argument("--max-frames-per-video", type=int, default=10)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any([args.vision, args.audio, args.all, args.extra_hockey, args.merge_youtube_violence, args.cctv_fights]):
        print("Select --vision, --audio, --extra-hockey, --merge-youtube-violence, or --all")
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
    if args.merge_youtube_violence:
        added = merge_youtube_violence_wavs(Path(args.violence_raw), Path(args.audio_out))
        print(f"Merged YouTube-violence WAVs into aggressive: {added} files attempted")

    if args.cctv_fights:
        prepare_cctv_fights_vision(
            cctv_root=Path(args.cctv_root),
            nonfight_dir=Path(args.nonfight_root),
            out_root=Path(args.vision_out),
            violent_frame_every=max(1, args.frame_every),
            violent_max_frames=max(1, args.max_frames_per_video),
            nonfight_frame_every=max(1, args.frame_every // 3 + 1),
            nonfight_max_frames=min(80, args.max_frames_per_video * 5),
            resize=max(1, args.resize),
        )

    print("Dataset preparation done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

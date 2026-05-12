from __future__ import annotations

import argparse
import csv as _csv
import shutil
import subprocess
import wave
from pathlib import Path

import cv2
import librosa
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}

# ESC-50 target class IDs → audio training class
ESC50_LABEL_MAP: dict[int, str] = {
    20: "distressed",       # crying_baby
    39: "physical_impacts", # glass_breaking
    30: "physical_impacts", # door_wood_knock (heavy locker-slam thuds)
    12: "physical_impacts", # crackling_fire (mic-clip proxy for physical hit)
    25: "neutral",          # footsteps
    24: "neutral",          # coughing
    21: "neutral",          # sneezing
    26: "neutral",          # laughing
    38: "neutral",          # clock_tick
}

# FSDKaggle2018 string labels → physical_impacts
FSD2018_IMPACT_LABELS = {"Shatter", "Drawer_open_or_close", "Bass_drum"}

ALL_AUDIO_CLASSES = ["aggressive", "distressed", "neutral", "physical_impacts"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_wav_mono(path: Path, y: np.ndarray, sr: int = 16000) -> None:
    y16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(y16.tobytes())


def _ffmpeg_to_wav(src: Path, dst: Path, sample_rate: int = 16000) -> bool:
    """Extract/convert any audio or video file to mono WAV via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        r = subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), str(dst)],
            capture_output=True,
            timeout=120,
        )
        return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def _chunk_audio_to_wavs(
    src: Path, out_dir: Path, prefix: str, chunk_sec: float = 2.0, sample_rate: int = 16000
) -> int:
    """Load src audio, split into non-overlapping chunks, save each as WAV. Returns chunk count."""
    try:
        y, _ = librosa.load(str(src), sr=sample_rate, mono=True)
    except Exception:
        return 0
    chunk_n = int(chunk_sec * sample_rate)
    saved = 0
    for i in range(len(y) // chunk_n):
        dst = out_dir / f"{prefix}_{i:04d}.wav"
        if not dst.exists():
            _write_wav_mono(dst, y[i * chunk_n:(i + 1) * chunk_n], sample_rate)
        saved += 1
    return saved


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
    for cls in ALL_AUDIO_CLASSES:
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


def prepare_vsd_audio(raw_root: Path, out_root: Path) -> None:
    """Copy Audio-based Violence Detection Dataset WAVs.

    Files named angry_*.wav → aggressive
    Files named noviolence_*.wav → neutral
    Searches raw_root recursively for WAV files with these prefixes.
    """
    for cls in ["aggressive", "neutral"]:
        ensure_dir(out_root / cls)

    copied_agg = copied_neu = 0
    for wav in raw_root.rglob("*.wav"):
        name = wav.name.lower()
        if name.startswith("angry"):
            dst = out_root / "aggressive" / f"vsd_{wav.name}"
            if not dst.exists():
                shutil.copy2(wav, dst)
            copied_agg += 1
        elif name.startswith("noviolence"):
            dst = out_root / "neutral" / f"vsd_{wav.name}"
            if not dst.exists():
                shutil.copy2(wav, dst)
            copied_neu += 1

    print(f"VSD audio prep: aggressive={copied_agg} neutral={copied_neu}")
    print(f"Output: {out_root}")


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


def prepare_esc50_audio(raw_root: Path, out_root: Path) -> None:
    """Copy ESC-50 clips into audio class folders using ESC50_LABEL_MAP.

    raw_root should contain the extracted ESC-50-master/ directory with
    meta/esc50.csv and audio/*.wav inside.
    """
    meta_csv = next(raw_root.rglob("esc50.csv"), None)
    if meta_csv is None:
        raise FileNotFoundError(f"esc50.csv not found under: {raw_root}")
    audio_dir = meta_csv.parent.parent / "audio"
    if not audio_dir.is_dir():
        audio_dir = next((p for p in raw_root.rglob("audio") if p.is_dir()), None)
        if audio_dir is None:
            raise FileNotFoundError(f"ESC-50 audio/ dir not found under: {raw_root}")

    # Handle nested audio/audio/ layout that some Kaggle downloads produce.
    if not list(audio_dir.glob("*.wav")):
        deeper = next(audio_dir.rglob("*.wav"), None)
        if deeper:
            audio_dir = deeper.parent

    for cls in ALL_AUDIO_CLASSES:
        ensure_dir(out_root / cls)

    copied = skipped = 0
    with meta_csv.open(encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            label = ESC50_LABEL_MAP.get(int(row["target"]))
            if label is None:
                continue
            src = audio_dir / row["filename"]
            if not src.exists():
                skipped += 1
                continue
            dst = out_root / label / f"esc50_{row['filename']}"
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1

    print(f"ESC-50 audio prep: copied={copied} skipped={skipped}")
    print(f"Output: {out_root}")


def prepare_fsd2018_audio(raw_root: Path, out_root: Path) -> None:
    """Copy FSDKaggle2018 physical-impact sounds into physical_impacts/.

    raw_root should contain train.csv and audio_train/*.wav extracted from the
    Freesound Audio Tagging competition download.
    Labels used: Shatter, Drawer_open_or_close, Bass_drum.
    """
    meta_csv = next(raw_root.rglob("train.csv"), None)
    if meta_csv is None:
        raise FileNotFoundError(f"train.csv not found under: {raw_root}")
    audio_dir = next((p for p in raw_root.rglob("audio_train") if p.is_dir()), None)
    if audio_dir is None:
        raise FileNotFoundError(f"audio_train/ dir not found under: {raw_root}")

    ensure_dir(out_root / "physical_impacts")
    copied = skipped = 0
    with meta_csv.open(encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get("label") not in FSD2018_IMPACT_LABELS:
                continue
            src = audio_dir / row["fname"]
            if not src.exists():
                skipped += 1
                continue
            dst = out_root / "physical_impacts" / f"fsd_{row['fname']}"
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1

    print(f"FSD2018 physical_impacts: copied={copied} skipped={skipped}")
    print(f"Output: {out_root}")


def prepare_xd_violence_audio(
    raw_root: Path,
    out_root: Path,
    max_per_folder: int = 500,
    sample_rate: int = 16000,
) -> None:
    """Extract audio tracks from XD-Violence mp4 clips.

    Folder mapping:
        Abuse/  → aggressive
        Riot/   → aggressive
        Fighting/ → physical_impacts
    """
    FOLDER_LABEL = {
        "abuse": "aggressive",
        "riot": "aggressive",
        "fighting": "physical_impacts",
    }
    for cls in ["aggressive", "physical_impacts"]:
        ensure_dir(out_root / cls)

    if not shutil.which("ffmpeg"):
        print("WARNING: ffmpeg not found — skipping XD-Violence audio extraction.")
        return

    total = 0
    for folder_key, label in FOLDER_LABEL.items():
        src_dirs = [p for p in raw_root.rglob("*") if p.is_dir() and p.name.lower() == folder_key]
        n = 0
        for src_dir in src_dirs:
            for vp in sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS):
                if n >= max_per_folder:
                    break
                dst = out_root / label / f"xdv_{folder_key}_{n:05d}.wav"
                if not dst.exists():
                    if not _ffmpeg_to_wav(vp, dst, sample_rate):
                        continue
                n += 1
        total += n
        print(f"  XD-Violence {folder_key!r} → {label}: {n} clips")

    print(f"XD-Violence audio prep complete. Total: {total}")
    print(f"Output: {out_root}")


def prepare_audio_noise_neutral(
    raw_root: Path,
    out_root: Path,
    chunk_sec: float = 2.0,
    sample_rate: int = 16000,
) -> None:
    """Chunk Audio Noise Dataset ambient recordings into 2-second neutral samples.

    Targets: sample-1.webm, sample-3.webm, sample-7.webm, sample-10.webm
    Each long recording is split into non-overlapping 2-second WAV chunks.
    """
    TARGET_FILES = {"sample-1.webm", "sample-10.webm", "sample-7.webm", "sample-3.webm"}
    ensure_dir(out_root / "neutral")
    total = 0
    for fname in TARGET_FILES:
        for src in raw_root.rglob(fname):
            prefix = f"noise_{src.stem}"
            n = _chunk_audio_to_wavs(src, out_root / "neutral", prefix, chunk_sec, sample_rate)
            total += n
            print(f"  {src.name}: {n} chunks → neutral")
    print(f"Audio Noise Dataset neutral: {total} chunks total")
    print(f"Output: {out_root}")


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
        "--vsd",
        action="store_true",
        help="Copy Audio-based Violence Detection Dataset (angry→aggressive, noviolence→neutral)",
    )
    parser.add_argument("--vsd-raw", default="dataset/raw/audio_violence_youtube", help="Extracted VSD root")
    parser.add_argument(
        "--merge-youtube-violence",
        action="store_true",
        help="After --audio, also copy Violence/*.wav from --violence-raw into audio/aggressive/",
    )
    parser.add_argument(
        "--esc50",
        action="store_true",
        help="Prepare ESC-50 audio clips (distressed/physical_impacts/neutral classes)",
    )
    parser.add_argument(
        "--fsd2018",
        action="store_true",
        help="Prepare FSDKaggle2018 physical-impact sounds (Shatter, Drawer_open_or_close, Bass_drum)",
    )
    parser.add_argument(
        "--xd-violence-audio",
        action="store_true",
        help="Extract audio from XD-Violence mp4 clips (requires ffmpeg)",
    )
    parser.add_argument(
        "--audio-noise",
        action="store_true",
        help="Chunk Audio Noise Dataset ambient webm files into neutral samples",
    )
    parser.add_argument("--esc50-raw", default="dataset/raw/esc50", help="Extracted ESC-50 root")
    parser.add_argument("--fsd2018-raw", default="dataset/raw/fsd2018", help="Extracted FSDKaggle2018 root")
    parser.add_argument("--xd-violence-raw", default="dataset/raw/xd_violence", help="Extracted XD-Violence root")
    parser.add_argument("--audio-noise-raw", default="dataset/raw/audio_noise", help="Extracted Audio Noise Dataset root")
    parser.add_argument("--xd-max-per-folder", type=int, default=500, help="Max XD-Violence clips per folder")

    parser.add_argument("--frame-every", type=int, default=45)
    parser.add_argument("--max-frames-per-video", type=int, default=10)
    parser.add_argument("--resize", type=int, default=224)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_flags = [args.audio, args.vsd, args.esc50, args.fsd2018, args.xd_violence_audio, args.audio_noise, args.merge_youtube_violence]
    if not any([args.vision, args.all, args.extra_hockey, args.cctv_fights, *audio_flags]):
        print("Select --vision, --audio, --esc50, --fsd2018, --xd-violence-audio, --audio-noise, "
              "--merge-youtube-violence, --extra-hockey, --cctv-fights, or --all")
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
    if args.vsd:
        prepare_vsd_audio(raw_root=Path(args.vsd_raw), out_root=Path(args.audio_out))

    if args.merge_youtube_violence:
        added = merge_youtube_violence_wavs(Path(args.violence_raw), Path(args.audio_out))
        print(f"Merged YouTube-violence WAVs into aggressive: {added} files attempted")

    if args.esc50:
        prepare_esc50_audio(raw_root=Path(args.esc50_raw), out_root=Path(args.audio_out))

    if args.fsd2018:
        prepare_fsd2018_audio(raw_root=Path(args.fsd2018_raw), out_root=Path(args.audio_out))

    if args.xd_violence_audio:
        prepare_xd_violence_audio(
            raw_root=Path(args.xd_violence_raw),
            out_root=Path(args.audio_out),
            max_per_folder=args.xd_max_per_folder,
        )

    if args.audio_noise:
        prepare_audio_noise_neutral(raw_root=Path(args.audio_noise_raw), out_root=Path(args.audio_out))

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

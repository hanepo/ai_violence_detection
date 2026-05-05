from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    "vision": {
        "slug": "mohamedmustafa/real-life-violence-situations-dataset",
        "target_dir": "dataset/raw/rwf2000",
        "description": "RWF-2000 style fight/non-fight video dataset",
    },
    "audio": {
        "slug": "ejlok1/cremad",
        "target_dir": "dataset/raw/cremad",
        "description": "CREMA-D emotion speech dataset",
    },
}


def run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "Unknown command error"
        raise RuntimeError(message)


def resolve_kaggle_cmd() -> list[str]:
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        return [kaggle_bin]
    return [sys.executable, "-m", "kaggle"]


def ensure_kaggle_ready() -> None:
    kaggle_cmd = resolve_kaggle_cmd()
    try:
        run_cmd(kaggle_cmd + ["--help"])
    except Exception as exc:
        raise RuntimeError("Kaggle CLI not found in current Python environment. Install with: pip install kaggle") from exc

    # Kaggle auth can be provided in multiple ways.
    if os.getenv("KAGGLE_API_TOKEN"):
        return

    has_env_auth = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    if has_env_auth:
        return

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        raise RuntimeError(
            "Missing Kaggle auth. Set KAGGLE_API_TOKEN, or set KAGGLE_USERNAME+KAGGLE_KEY, or add ~/.kaggle/kaggle.json."
        )


def unzip_all(zip_files: list[Path], out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for zf in zip_files:
        with zipfile.ZipFile(zf, "r") as zip_ref:
            for member in zip_ref.infolist():
                dest = out_dir / member.filename
                if dest.exists() and not force:
                    continue
                zip_ref.extract(member, out_dir)


def download_dataset(kind: str, force: bool, skip_unzip: bool) -> None:
    spec = DATASETS[kind]
    raw_dir = Path("dataset/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    target_dir = Path(spec["target_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {kind}: {spec['description']}")
    print(f"Kaggle slug: {spec['slug']}")

    cmd = resolve_kaggle_cmd() + [
        "datasets",
        "download",
        "-d",
        spec["slug"],
        "-p",
        str(target_dir),
    ]
    if force:
        cmd.append("--force")

    run_cmd(cmd)

    # Remove empty argument if --force is not used.
    for p in list(target_dir.glob("*.zip")):
        if skip_unzip:
            print(f"Downloaded zip: {p}")
            continue
        print(f"Extracting: {p}")
        unzip_all([p], target_dir, force=force)

    print(f"Done: {target_dir}")


def download_kaggle_slug(slug: str, target_dir: Path, force: bool, skip_unzip: bool) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading custom Kaggle dataset: {slug}")

    cmd = resolve_kaggle_cmd() + [
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(target_dir),
    ]
    if force:
        cmd.append("--force")
    run_cmd(cmd)

    for p in list(target_dir.glob("*.zip")):
        if skip_unzip:
            print(f"Downloaded zip: {p}")
            continue
        print(f"Extracting: {p}")
        unzip_all([p], target_dir, force=force)

    print(f"Done: {target_dir}")


def download_zip_url(zip_url: str, target_dir: Path, force: bool, skip_unzip: bool) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = zip_url.rstrip("/").split("/")[-1] or "dataset.zip"
    if not filename.endswith(".zip"):
        filename = f"{filename}.zip"
    zip_path = target_dir / filename

    if zip_path.exists() and not force:
        print(f"Zip already exists: {zip_path}")
    else:
        print(f"Downloading zip URL: {zip_url}")
        urllib.request.urlretrieve(zip_url, str(zip_path))

    if skip_unzip:
        print(f"Downloaded zip: {zip_path}")
    else:
        print(f"Extracting: {zip_path}")
        unzip_all([zip_path], target_dir, force=force)

    print(f"Done: {target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-download baseline training datasets (RWF-2000 and CREMA-D)."
    )
    parser.add_argument("--vision", action="store_true", help="Download vision dataset")
    parser.add_argument("--audio", action="store_true", help="Download audio dataset")
    parser.add_argument("--all", action="store_true", help="Download both datasets")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    parser.add_argument("--skip-unzip", action="store_true", help="Keep zip files only")
    parser.add_argument("--kaggle-slug", type=str, default=None, help="Custom Kaggle slug owner/dataset")
    parser.add_argument("--zip-url", type=str, default=None, help="Direct ZIP URL for non-Kaggle datasets")
    parser.add_argument("--target-dir", type=str, default="dataset/raw/custom", help="Target output directory for custom downloads")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any([args.vision, args.audio, args.all, args.kaggle_slug, args.zip_url]):
        print("Select --vision, --audio, --all, --kaggle-slug, or --zip-url")
        return 1

    try:
        if args.vision or args.audio or args.all or args.kaggle_slug:
            ensure_kaggle_ready()
        if args.all or args.vision:
            download_dataset("vision", force=args.force, skip_unzip=args.skip_unzip)
        if args.all or args.audio:
            download_dataset("audio", force=args.force, skip_unzip=args.skip_unzip)
        if args.kaggle_slug:
            download_kaggle_slug(
                slug=args.kaggle_slug,
                target_dir=Path(args.target_dir),
                force=args.force,
                skip_unzip=args.skip_unzip,
            )
        if args.zip_url:
            download_zip_url(
                zip_url=args.zip_url,
                target_dir=Path(args.target_dir),
                force=args.force,
                skip_unzip=args.skip_unzip,
            )
    except Exception as exc:
        print(f"Download failed: {exc}")
        return 2

    print("All requested dataset downloads completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
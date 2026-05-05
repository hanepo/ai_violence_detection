from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Allow running this file from either project root or scripts/ directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from utils.config import load_settings
from utils.video import VideoCaptureStream


def run_preview(save_dir: Path, max_frames: int = 300) -> None:
	settings = load_settings()
	save_dir.mkdir(parents=True, exist_ok=True)

	stream = VideoCaptureStream(
		camera_index=settings.camera_index,
		width=settings.frame_width,
		height=settings.frame_height,
		fps=settings.fps,
	)

	frame_count = 0
	try:
		while frame_count < max_frames:
			frame, _ = stream.read()
			if frame is None:
				continue

			cv2.imshow("Camera Preview", frame)
			if frame_count % 30 == 0:
				out = save_dir / f"frame_{frame_count:05d}.jpg"
				cv2.imwrite(str(out), frame)

			frame_count += 1
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break
	finally:
		stream.release()
		cv2.destroyAllWindows()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Camera bring-up and preview script")
	parser.add_argument("--save-dir", default="output/camera_test")
	parser.add_argument("--max-frames", type=int, default=300)
	args = parser.parse_args()
	run_preview(Path(args.save_dir), args.max_frames)

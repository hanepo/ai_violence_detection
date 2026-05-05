from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow running this file from either project root or scripts/ directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from utils.config import load_settings
from utils.inference import VisionInference


def detect_image(image_path: Path) -> None:
	settings = load_settings()
	model = VisionInference(settings.vision_model_path)

	frame = cv2.imread(str(image_path))
	if frame is None:
		raise FileNotFoundError(f"Unable to read image: {image_path}")

	score = model.score(frame)
	label = "violent-risk" if score > settings.threshold else "normal"
	print(f"image={image_path} score={score:.4f} label={label}")


def _seconds_to_hhmmss(sec: float) -> str:
	sec = max(0.0, sec)
	h = int(sec // 3600)
	m = int((sec % 3600) // 60)
	s = sec % 60
	return f"{h:02d}:{m:02d}:{s:05.2f}"


def _find_risk_segments(points: list[tuple[float, float]], threshold: float, max_gap_seconds: float = 1.2) -> list[tuple[float, float]]:
	risk_times = [t for t, score in points if score > threshold]
	if not risk_times:
		return []

	segments: list[tuple[float, float]] = []
	start = risk_times[0]
	prev = risk_times[0]

	for t in risk_times[1:]:
		if (t - prev) <= max_gap_seconds:
			prev = t
			continue
		segments.append((start, prev))
		start = t
		prev = t

	segments.append((start, prev))
	return segments


def detect_video(
	video_path: Path,
	sample_every: int,
	threshold: float,
	report_json: Path | None,
	annotate_out: Path | None,
) -> None:
	settings = load_settings()
	model = VisionInference(settings.vision_model_path)
	cap = cv2.VideoCapture(str(video_path))
	fps = cap.get(cv2.CAP_PROP_FPS) or float(settings.fps)

	writer = None
	if annotate_out is not None:
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or settings.frame_width)
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or settings.frame_height)
		annotate_out.parent.mkdir(parents=True, exist_ok=True)
		writer = cv2.VideoWriter(str(annotate_out), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (w, h))

	i = 0
	scores = []
	timeline: list[tuple[float, float]] = []
	while True:
		ok, frame = cap.read()
		if not ok:
			break

		t = i / max(fps, 1.0)
		score = None
		if i % sample_every == 0:
			score = model.score(frame)
			scores.append(score)
			timeline.append((t, score))

		if writer is not None:
			display_score = score if score is not None else (scores[-1] if scores else 0.0)
			label = "violent-risk" if display_score > threshold else "normal"
			color = (0, 0, 255) if label == "violent-risk" else (0, 200, 0)
			cv2.putText(frame, f"score={display_score:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
			cv2.putText(frame, label, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
			cv2.putText(frame, _seconds_to_hhmmss(t), (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
			writer.write(frame)
		i += 1

	cap.release()
	if writer is not None:
		writer.release()

	if not scores:
		print("No frames processed")
		return

	avg_score = sum(scores) / len(scores)
	max_score = max(scores)
	label = "violent-risk" if avg_score > threshold else "normal"
	segments = _find_risk_segments(timeline, threshold=threshold)

	print(f"video={video_path}")
	print(f"samples={len(scores)} avg_score={avg_score:.4f} max_score={max_score:.4f} threshold={threshold:.3f}")
	print(f"video_label={label}")

	if segments:
		print("risk_segments:")
		for start, end in segments:
			print(f"  - {_seconds_to_hhmmss(start)} -> {_seconds_to_hhmmss(end)}")
	else:
		print("risk_segments: none")

	if annotate_out is not None:
		print(f"annotated_video={annotate_out}")

	if report_json is not None:
		report_json.parent.mkdir(parents=True, exist_ok=True)
		report = {
			"video": str(video_path),
			"samples": len(scores),
			"avg_score": float(avg_score),
			"max_score": float(max_score),
			"threshold": float(threshold),
			"video_label": label,
			"risk_segments_seconds": [[float(s), float(e)] for s, e in segments],
		}
		report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
		print(f"report_json={report_json}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run basic detection on image/video")
	parser.add_argument("input", type=str, help="Path to image or video")
	parser.add_argument("--sample-every", type=int, default=5)
	parser.add_argument("--threshold", type=float, default=None, help="Override detection threshold")
	parser.add_argument("--report-json", type=str, default=None, help="Optional path to save JSON summary")
	parser.add_argument("--annotate-out", type=str, default=None, help="Optional path to save annotated mp4")
	args = parser.parse_args()

	path = Path(args.input)
	suffix = path.suffix.lower()
	if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
		detect_image(path)
	else:
		settings = load_settings()
		threshold = settings.threshold if args.threshold is None else float(args.threshold)
		detect_video(
			path,
			sample_every=max(1, args.sample_every),
			threshold=threshold,
			report_json=Path(args.report_json) if args.report_json else None,
			annotate_out=Path(args.annotate_out) if args.annotate_out else None,
		)

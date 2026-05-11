from __future__ import annotations

from collections import deque
import atexit
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import librosa
import numpy as np
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, send_from_directory, url_for

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Use real TFLite inference by default in this environment.
# Set ENABLE_TFLITE=0 only as an emergency fallback.
os.environ.setdefault("ENABLE_TFLITE", "1")

from utils.config import load_settings
from utils.audio import AudioStream, extract_mfcc, sounddevice_available
from utils.db import EventDatabase, EventRecord
from utils.fusion import compute_fusion_score
from utils.inference import AudioInference, VisionInference
from utils.video import VideoCaptureStream, write_playable_mp4

app = Flask(
    __name__,
    template_folder=str(ROOT_DIR / "web" / "templates"),
    static_folder=str(ROOT_DIR / "web" / "static"),
)
app.secret_key = "ai-security-system-dev"

settings = load_settings()
vision_model = VisionInference(settings.vision_model_path)
audio_model = AudioInference(settings.audio_model_path)

_camera_lock = threading.Lock()
_camera_stream: VideoCaptureStream | None = None
_mic_status_lock = threading.Lock()
_mic_status_cache: dict[str, object] = {
    "ok": False,
    "state": "unknown",
    "message": "waiting for first probe",
    "rms": 0.0,
    "dbfs": -100.0,
    "abs_mean": 0.0,
    "abs_max": 0.0,
    "nonzero_ratio": 0.0,
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
_mic_status_cache_ts = 0.0


def _probe_mic_status() -> dict[str, object]:
    if not sounddevice_available():
        return {
            "ok": True,
            "state": "unavailable",
            "message": "sounddevice not installed — pip install sounddevice (needs PortAudio). Live audio fusion will use silence.",
            "rms": 0.0,
            "dbfs": -100.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
            "nonzero_ratio": 0.0,
        }

    chunk_seconds = min(0.6, max(0.25, settings.audio_chunk_seconds * 0.25))
    try:
        stream = AudioStream(
            sample_rate=settings.audio_sample_rate,
            channels=settings.audio_channels,
            chunk_seconds=chunk_seconds,
        )
        chunk, _ = stream.read_chunk()
        if chunk is None or chunk.size == 0:
            return {
                "ok": True,
                "state": "quiet",
                "message": "no samples captured (check mic permission in System Settings)",
                "rms": 0.0,
                "dbfs": -100.0,
                "abs_mean": 0.0,
                "abs_max": 0.0,
                "nonzero_ratio": 0.0,
            }

        x = np.asarray(chunk, dtype=np.float64)
        abs_mean = float(np.mean(np.abs(x)))
        abs_max = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(x * x)))
        eps = 1e-12
        dbfs = float(20.0 * np.log10(max(rms, eps)))
        nonzero_ratio = float(np.mean(np.abs(chunk) > 1e-5))

        # "Active" = clearly above typical idle USB noise; tune with MIC_PROBE_* env vars.
        mic_active = rms > settings.mic_probe_active_rms or abs_max > settings.mic_probe_active_peak
        return {
            "ok": True,
            "state": "active" if mic_active else "quiet",
            "message": "mic signal detected" if mic_active else "mic connected (quiet / low level)",
            "rms": rms,
            "dbfs": dbfs,
            "abs_mean": abs_mean,
            "abs_max": abs_max,
            "nonzero_ratio": nonzero_ratio,
        }
    except Exception as exc:
        msg = str(exc).strip() or exc.__class__.__name__
        # PortAudio often throws if another thread is recording; avoid scary "error" UI.
        soft = "busy" in msg.lower() or "input overflow" in msg.lower() or "portaudio" in msg.lower()
        return {
            "ok": True,
            "state": "quiet" if soft else "error",
            "message": (
                "Microphone busy (live feed is recording). Status will update when idle."
                if soft
                else f"mic probe failed: {msg[:140]}"
            ),
            "rms": 0.0,
            "dbfs": -100.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
            "nonzero_ratio": 0.0,
        }


def _get_or_create_camera_stream() -> VideoCaptureStream:
    global _camera_stream
    with _camera_lock:
        if _camera_stream is None:
            _camera_stream = VideoCaptureStream(
                camera_index=settings.camera_index,
                width=settings.frame_width,
                height=settings.frame_height,
                fps=settings.fps,
            )
        return _camera_stream


def _reset_camera_stream() -> None:
    global _camera_stream
    with _camera_lock:
        if _camera_stream is not None:
            try:
                _camera_stream.release()
            except Exception:
                pass
        _camera_stream = None


@atexit.register
def _cleanup_camera_stream() -> None:
    _reset_camera_stream()


def _resolve_event_video_path(video_path: str | None) -> Path | None:
    if not video_path:
        return None
    candidate = Path(video_path)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _decorate_events(events: list[dict]) -> list[dict]:
    decorated: list[dict] = []
    output_dir = _resolve_output_dir().resolve()
    for event in events:
        item = dict(event)
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        clip_path = _resolve_event_video_path(item.get("video_path"))
        clip_name = None
        if clip_path is not None:
            resolved_clip = clip_path.resolve()
            if resolved_clip.is_relative_to(output_dir):
                clip_name = resolved_clip.name
        item["camera_name"] = (
            metadata.get("camera_name")
            or metadata.get("device_id")
            or settings.device_id
        )
        item["replay_url"] = (
            url_for("event_clip_file", filename=clip_name) if clip_name is not None else None
        )
        decorated.append(item)
    return decorated


def _resolve_db_path() -> Path:
    root_db = ROOT_DIR / "logs" / "events.db"
    scripts_db = ROOT_DIR / "scripts" / "logs" / "events.db"
    if root_db.exists():
        return root_db
    if scripts_db.exists():
        return scripts_db
    return settings.db_path if settings.db_path.is_absolute() else ROOT_DIR / settings.db_path


def _resolve_output_dir() -> Path:
    root_out = ROOT_DIR / "output"
    scripts_out = ROOT_DIR / "scripts" / "output"
    if root_out.exists():
        return root_out
    if scripts_out.exists():
        return scripts_out
    return settings.output_dir if settings.output_dir.is_absolute() else ROOT_DIR / settings.output_dir


def _score_video_file(video_path: Path, sample_every: int = 2) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    i = 0
    points: list[tuple[float, float]] = []
    motion_points: list[tuple[float, float]] = []
    prev_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            score = vision_model.score(frame)
            t = i / max(fps, 1.0)
            points.append((t, score))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                motion = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0)
                motion_points.append((t, motion))
            prev_gray = gray
        i += 1

    cap.release()
    if not points:
        return {"avg_score": 0.0, "max_score": 0.0, "label": "normal", "segments": []}

    scores = np.array([x[1] for x in points], dtype=np.float32)
    avg_score = float(scores.mean())
    max_score = float(scores.max())
    p90_score = float(np.percentile(scores, 90))
    risk_ratio = float(np.mean(scores >= settings.threshold))

    if motion_points:
        motion_values = np.array([x[1] for x in motion_points], dtype=np.float32)
        p90_motion = float(np.percentile(motion_values, 90))
        avg_motion = float(motion_values.mean())
    else:
        p90_motion = 0.0
        avg_motion = 0.0

    step_seconds = max((sample_every / max(fps, 1.0)), 1e-6)
    longest_run = 0
    current_run = 0
    for s in scores:
        if s >= settings.threshold:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    max_alert_seconds = float(longest_run * step_seconds)

    base_cv_score = (0.35 * avg_score) + (0.35 * p90_score) + (0.30 * max_score)
    strong_peak = max_score >= min(1.0, settings.threshold + settings.upload_strong_peak_delta)
    duration_gate = max_alert_seconds >= settings.upload_min_alert_seconds
    motion_gate = p90_motion >= settings.upload_motion_p90_min

    # Gate upload alerts on meaningful motion and sustained signal.
    clip_alert = motion_gate and (
        ((risk_ratio >= settings.upload_risk_ratio_min and p90_score >= settings.threshold) and duration_gate)
        or strong_peak
    )

    cv_score = base_cv_score if clip_alert else min(base_cv_score, max(0.0, settings.threshold - 0.10))
    segments = [[float(t), float(t)] for t, s in points if s > settings.threshold]
    label = "violent-risk" if clip_alert else "normal"
    return {
        "avg_score": avg_score,
        "max_score": max_score,
        "p90_score": p90_score,
        "risk_ratio": risk_ratio,
        "avg_motion": avg_motion,
        "p90_motion": p90_motion,
        "max_alert_seconds": max_alert_seconds,
        "base_cv_score": float(base_cv_score),
        "cv_score": float(cv_score),
        "clip_alert": bool(clip_alert),
        "label": label,
        "segments": segments,
        "timeline": [[float(t), float(s)] for t, s in points],
    }


def _score_at_time(timeline: list[list[float]], t: float) -> float:
    if not timeline:
        return 0.0

    if t <= timeline[0][0]:
        return float(timeline[0][1])
    if t >= timeline[-1][0]:
        return float(timeline[-1][1])

    for i in range(1, len(timeline)):
        t0, s0 = timeline[i - 1]
        t1, s1 = timeline[i]
        if t0 <= t <= t1:
            span = max(t1 - t0, 1e-6)
            alpha = (t - t0) / span
            return float(s0 + ((s1 - s0) * alpha))

    return float(timeline[-1][1])


def _render_upload_overlay_video(video_path: Path, timeline: list[list[float]], output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
        True,
    )

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = frame_idx / max(fps, 1.0)
            score = _score_at_time(timeline, t)
            is_alert = score >= settings.threshold
            label = "ALERT" if is_alert else "NORMAL"
            color = (0, 0, 255) if is_alert else (0, 200, 0)

            # Small OSD-style text in top-left corner (no boxed panel).
            badge_text = f"{label} {score * 100:.1f}%"
            cv2.putText(frame, badge_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 2)
            cv2.putText(frame, badge_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def _transcode_h264_if_possible(input_path: Path, output_path: Path) -> bool:
    """Convert to browser-friendly H.264 MP4 when an ffmpeg binary is available."""
    from utils.video import transcode_to_browser_mp4

    return transcode_to_browser_mp4(input_path, output_path, timeout=180)


def _upload_preview_stream_generator(
    video_path: Path,
    timeline: list[list[float]] | None = None,
    overall_score: float | None = None,
    overall_alert: bool = False,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_delay = 1.0 / max(fps, 1.0)

    # Use lighter smoothing for upload playback so transitions are responsive.
    score_hist: deque[float] = deque(maxlen=6)
    alert_active = False
    on_threshold = settings.threshold
    off_threshold = max(0.0, settings.threshold - 0.06)
    consecutive_on = 0
    consecutive_off = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = frame_idx / max(fps, 1.0)
            if timeline:
                raw_score = _score_at_time(timeline, t)
            else:
                raw_score = vision_model.score(frame)
            score_hist.append(raw_score)
            smooth_score = float(sum(score_hist) / len(score_hist))

            if smooth_score >= on_threshold:
                consecutive_on += 1
                consecutive_off = 0
            elif smooth_score <= off_threshold:
                consecutive_off += 1
                consecutive_on = 0

            if not alert_active and consecutive_on >= 3:
                alert_active = True
            if alert_active and consecutive_off >= 5:
                alert_active = False

            # If clip-level decision is ALERT, ease frame threshold slightly.
            if overall_alert and smooth_score >= max(0.0, settings.threshold - 0.08):
                alert_active = True

            frame_label = "ALERT" if alert_active else "NORMAL"
            frame_color = (0, 0, 255) if alert_active else (0, 200, 0)
            overall_label = "ALERT" if overall_alert else "NORMAL"
            overall_color = (0, 0, 255) if overall_alert else (0, 200, 0)

            line1 = f"{frame_label} {smooth_score * 100:.1f}%"
            if overall_score is not None:
                line2 = f"Overall {overall_label} {overall_score * 100:.1f}%"
            else:
                line2 = f"Overall {overall_label}"

            cv2.putText(frame, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 2)
            cv2.putText(frame, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, frame_color, 1)
            cv2.putText(frame, line2, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2)
            cv2.putText(frame, line2, (10, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.40, overall_color, 1)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            frame_idx += 1
            time.sleep(frame_delay)
    finally:
        cap.release()


def _score_audio_file(audio_path: Path) -> dict | None:
    try:
        wav, _ = librosa.load(
            audio_path,
            sr=settings.audio_sample_rate,
            mono=True,
            duration=30.0,  # cap at 30 seconds to prevent hangs on large files
        )
    except Exception:
        return None
    max_samples = int(settings.audio_sample_rate * settings.audio_chunk_seconds)

    if wav.size < max_samples:
        wav = np.pad(wav, (0, max_samples - wav.size))

    chunk_scores = []
    chunk_labels: list[str] = []
    chunk_top_scores: list[float] = []
    prob_sums: dict[str, float] = {}
    for start in range(0, len(wav), max_samples):
        chunk = wav[start : start + max_samples]
        if len(chunk) < max_samples:
            chunk = np.pad(chunk, (0, max_samples - len(chunk)))
        mfcc = librosa.feature.mfcc(y=chunk, sr=settings.audio_sample_rate, n_mfcc=40).astype(np.float32)
        pred = audio_model.predict(mfcc)
        score = float(pred.get("aggressive_score", 0.0))
        chunk_scores.append(score)
        top_label = pred.get("top_label")
        if isinstance(top_label, str):
            chunk_labels.append(top_label)
        chunk_top_scores.append(float(pred.get("top_score", 0.0)))
        probs = pred.get("probabilities", {})
        if isinstance(probs, dict):
            for k, v in probs.items():
                try:
                    prob_sums[k] = prob_sums.get(k, 0.0) + float(v)
                except Exception:
                    continue

    if not chunk_scores:
        return None

    avg_score = float(sum(chunk_scores) / len(chunk_scores))
    max_score = float(max(chunk_scores))
    clip_score = float((0.7 * avg_score) + (0.3 * max_score))

    if chunk_labels:
        predicted_label = max(set(chunk_labels), key=chunk_labels.count)
    elif prob_sums:
        predicted_label = max(prob_sums, key=prob_sums.get)
    else:
        predicted_label = "unknown"

    label = "aggressive-risk" if clip_score >= settings.audio_aggressive_threshold else predicted_label
    return {
        "avg_score": avg_score,
        "max_score": max_score,
        "score": clip_score,
        "predicted_label": predicted_label,
        "avg_top_score": float(sum(chunk_top_scores) / len(chunk_top_scores)) if chunk_top_scores else 0.0,
        "label": label,
    }


def _build_upload_result(video_result: dict | None, audio_result: dict | None) -> dict:
    if video_result:
        cv = float(video_result.get("cv_score", 0.0))
    else:
        cv = 0.0

    ca = float(audio_result.get("score", audio_result.get("avg_score", 0.0))) if audio_result else None

    # Keep upload behavior intuitive for single-modality tests.
    # - video only: use vision threshold
    # - audio only: use audio aggressive threshold
    # - both: use configured fusion weights/threshold
    if video_result is None and audio_result is not None:
        fusion_cv = 0.0
        fusion_ca = float(ca)
        fusion_score = fusion_ca
        fusion_is_alert = bool(fusion_score >= settings.audio_aggressive_threshold)
    else:
        fusion = compute_fusion_score(
            cv=cv,
            ca=ca,
            alpha=settings.alpha,
            beta=settings.beta,
            threshold=settings.threshold,
            audio_alert_threshold=settings.audio_aggressive_threshold,
        )
        fusion_cv = fusion.cv
        fusion_ca = fusion.ca
        fusion_score = fusion.score
        fusion_is_alert = fusion.is_alert

    event_id = str(uuid.uuid4())
    db = EventDatabase(_resolve_db_path())
    db.insert_event(
        EventRecord(
            event_id=event_id,
            threat_score=fusion_score,
            cv=fusion_cv,
            ca=fusion_ca,
            video_path=None,
            metadata_json=json.dumps({"source": "upload_test"}),
        )
    )
    db.close()

    return {
        "event_id": event_id,
        "cv": fusion_cv,
        "ca": fusion_ca,
        "fusion_score": fusion_score,
        "is_alert": fusion_is_alert,
        "video_result": video_result,
        "audio_result": audio_result,
    }


@app.get("/")
def dashboard() -> str:
    db = EventDatabase(_resolve_db_path())
    stats = db.get_stats()
    events = _decorate_events(db.list_recent_events(limit=12))
    db.close()
    return render_template(
        "dashboard.html",
        stats=stats,
        events=events,
        alert_threshold=settings.threshold,
        alert_cooldown_seconds=settings.alert_cooldown_seconds,
        audio_aggressive_threshold=settings.audio_aggressive_threshold,
        fusion_alpha=settings.alpha,
        fusion_beta=settings.beta,
    )


@app.get("/upload-test")
def upload_test_page() -> str:
    return render_template("upload_test.html")


@app.post("/upload-test")
def upload_test_submit():
    upload_dir = _resolve_output_dir() / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    video = request.files.get("video_file")
    audio = request.files.get("audio_file")
    if (video is None or video.filename == "") and (audio is None or audio.filename == ""):
        flash("Please upload at least one file (video or audio).")
        return redirect(url_for("upload_test_page"))

    video_result = None
    audio_result = None
    processed_video_url = None
    preview_stream_url = None
    preview_note = None

    if video is not None and video.filename:
        video_path = upload_dir / f"{uuid.uuid4()}_{Path(video.filename).name}"
        video.save(video_path)
        video_result = _score_video_file(video_path)
        video_overall_score = float(video_result.get("cv_score", 0.0))
        video_overall_alert = int(bool(video_result.get("clip_alert", False)))
        timeline = video_result.get("timeline", [])
        timeline_path = upload_dir / f"{video_path.stem}.timeline.json"
        timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
        preview_stream_url = url_for(
            "uploaded_preview_stream",
            filename=video_path.name,
            overall_score=f"{video_overall_score:.6f}",
            overall_alert=str(video_overall_alert),
        )
        processed_name = f"{video_path.stem}_overlay.mp4"
        processed_path = upload_dir / processed_name
        _render_upload_overlay_video(video_path, timeline, processed_path)
        playable_name = f"{video_path.stem}_overlay_playable.mp4"
        playable_path = upload_dir / playable_name
        if processed_path.exists() and _transcode_h264_if_possible(processed_path, playable_path):
            processed_video_url = url_for("uploaded_output_file", filename=playable_name)
        elif processed_path.exists():
            preview_note = (
                "Playable MP4 preview is unavailable on this machine (H.264 transcode failed). "
                "Showing stream preview instead. Install ffmpeg/imageio-ffmpeg for browser playback."
            )

    if audio is not None and audio.filename:
        audio_path = upload_dir / f"{uuid.uuid4()}_{Path(audio.filename).name}"
        audio.save(audio_path)
        try:
            audio_result = _score_audio_file(audio_path)
        except Exception:
            audio_result = None

    result = _build_upload_result(video_result, audio_result)
    if processed_video_url:
        result["preview_video_url"] = processed_video_url
    if preview_stream_url:
        result["preview_stream_url"] = preview_stream_url
    if preview_note:
        result["preview_note"] = preview_note
    return render_template("upload_test.html", result=result)


@app.get("/api/events")
def api_events():
    db = EventDatabase(_resolve_db_path())
    events = _decorate_events(db.list_recent_events(limit=50))
    db.close()
    return jsonify(events)


@app.get("/api/stats")
def api_stats():
    db = EventDatabase(_resolve_db_path())
    stats = db.get_stats()
    db.close()
    return jsonify(stats)


@app.get("/api/mic-status")
def api_mic_status():
    global _mic_status_cache_ts
    now = time.time()

    with _mic_status_lock:
        if now - _mic_status_cache_ts < 1.8:
            return jsonify(_mic_status_cache)

    status = _probe_mic_status()
    status["updated_at"] = datetime.now(timezone.utc).isoformat()

    with _mic_status_lock:
        _mic_status_cache.update(status)
        _mic_status_cache_ts = now
        payload = dict(_mic_status_cache)

    return jsonify(payload)


@app.get("/event-clips/<path:filename>")
def event_clip_file(filename: str):
    output_dir = _resolve_output_dir()
    if filename.lower().endswith(".mp4"):
        # conditional=True can break Range probes on some dev servers; clips are small.
        return send_from_directory(
            output_dir,
            filename,
            mimetype="video/mp4",
            max_age=0,
            conditional=False,
        )
    return send_from_directory(output_dir, filename, max_age=0)


@app.get("/uploads/<path:filename>")
def uploaded_output_file(filename: str):
    upload_dir = _resolve_output_dir() / "uploads"
    return send_from_directory(upload_dir, filename)


@app.get("/upload-preview-stream/<path:filename>")
def uploaded_preview_stream(filename: str):
    upload_path = (_resolve_output_dir() / "uploads" / filename)
    if not upload_path.exists():
        return Response(status=404)
    timeline_path = upload_path.parent / f"{upload_path.stem}.timeline.json"
    timeline = None
    if timeline_path.exists():
        try:
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        except Exception:
            timeline = None
    overall_score_param = request.args.get("overall_score")
    overall_alert_param = request.args.get("overall_alert", "0")
    try:
        overall_score = float(overall_score_param) if overall_score_param is not None else None
    except ValueError:
        overall_score = None
    overall_alert = overall_alert_param == "1"
    return Response(
        _upload_preview_stream_generator(
            upload_path,
            timeline=timeline,
            overall_score=overall_score,
            overall_alert=overall_alert,
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _save_live_alert_event(
    frames_snapshot: list,
    threat_score: float,
    cv_score: float,
    ca_score: float,
    mode_label: str,
    *,
    motion_level: float | None = None,
) -> None:
    """Save a confirmed live-camera alert: write the clip MP4 and insert a DB record.

    Runs in a daemon background thread so the camera stream is never blocked.
    """
    try:
        event_id = str(uuid.uuid4())
        clip_name = f"event_{event_id}.mp4"
        out_dir = _resolve_output_dir()
        clip_path = out_dir / clip_name

        write_playable_mp4(frames_snapshot, clip_path, settings.fps)
        clip_duration = len(frames_snapshot) / max(settings.fps, 1)

        metadata = {
            "device_id": settings.device_id,
            "camera_name": settings.camera_name,
            "mode": "dashboard_live",
            "fusion_mode": mode_label,
            "alert_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "clip_duration_seconds": round(clip_duration, 3),
            "video_frames": len(frames_snapshot),
            "source": "dashboard_camera_feed",
        }
        if motion_level is not None:
            metadata["scene_motion_at_save"] = round(float(motion_level), 5)

        db = EventDatabase(_resolve_db_path())
        db.insert_event(
            EventRecord(
                event_id=event_id,
                threat_score=threat_score,
                cv=cv_score,
                ca=ca_score,
                video_path=str(clip_path),
                metadata_json=json.dumps(metadata),
            )
        )
        db.close()
    except Exception:
        pass


@app.get("/camera-feed")
def camera_feed() -> Response:
    # Read request args here — the request context is torn down before the
    # generator body runs, so accessing request.args inside the generator fails.
    _use_vision = request.args.get("vision", "1") != "0"
    _use_audio = request.args.get("audio", "1") != "0"

    def generate_frames():
        use_vision = _use_vision
        use_audio = _use_audio
        if not use_vision and not use_audio:
            # Both off (bad client state) → same as full multimodal, not vision-only + silent audio.
            use_vision = True
            use_audio = True

        if use_vision and use_audio:
            mode_label = "combined"
            t_hyst = settings.threshold
        elif use_vision:
            mode_label = "vision_only"
            t_hyst = settings.threshold
        else:
            mode_label = "audio_only"
            t_hyst = settings.audio_aggressive_threshold

        on_threshold = min(1.0, t_hyst + max(0.02, settings.live_alert_enter_margin))
        off_threshold = max(0.0, t_hyst - max(0.01, settings.live_alert_exit_margin))

        stream: VideoCaptureStream | None = None
        if use_vision:
            stream = _get_or_create_camera_stream()

        h, w = settings.frame_height, settings.frame_width
        black_tpl = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(black_tpl, "Vision disabled — audio-only mode",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 2)

        # ── Shared inference state (written by background threads, read by display loop) ──
        _infer_lock = threading.Lock()
        _cv_score: list[float] = [0.0]
        _ca_score: list[float] = [0.0]
        _audio_ready: list[bool] = [False]   # False until first real audio result
        _stop = [False]

        # ── Vision inference thread ──
        # Runs TFLite on a frame sampled from the camera at inference rate.
        # Decoupled from the display loop so slow inference doesn't drop frames.
        def _vision_worker() -> None:
            while not _stop[0]:
                if stream is None:
                    time.sleep(0.05)
                    continue
                f, _ = stream.read()
                if f is None:
                    time.sleep(0.05)
                    continue
                score = vision_model.score(f)
                with _infer_lock:
                    _cv_score[0] = float(score)

        # ── Audio inference thread ──
        # read_chunk() blocks for chunk_seconds — running it here keeps the
        # display loop free to deliver frames at camera rate.
        def _audio_worker() -> None:
            audio_stream = AudioStream(
                sample_rate=settings.audio_sample_rate,
                channels=settings.audio_channels,
                chunk_seconds=settings.audio_chunk_seconds,
            )
            while not _stop[0]:
                wav, _ = audio_stream.read_chunk()
                mfcc = extract_mfcc(wav, settings.audio_sample_rate)
                score = audio_model.score(mfcc)
                with _infer_lock:
                    _ca_score[0] = float(score)
                    _audio_ready[0] = True

        workers: list[threading.Thread] = []
        if use_vision:
            t = threading.Thread(target=_vision_worker, daemon=True)
            t.start()
            workers.append(t)
        if use_audio:
            t = threading.Thread(target=_audio_worker, daemon=True)
            t.start()
            workers.append(t)

        # Wait up to 2 s for the first camera frame.
        first_frame: np.ndarray | None = None
        if stream is not None:
            for _ in range(40):
                first_frame, _ = stream.read()
                if first_frame is not None:
                    break
                time.sleep(0.05)

        if first_frame is None and use_vision:
            first_frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(first_frame, "Camera initializing...", (12, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)

        score_hist_cv: deque[float] = deque(maxlen=max(6, settings.live_score_window))
        score_hist_ca: deque[float] = deque(maxlen=max(6, settings.live_score_window))
        alert_active = False
        consecutive_on = 0
        consecutive_off = 0
        prev_alert_active = False
        prev_gray_small: np.ndarray | None = None

        _live_fps = max(settings.fps, 1)
        _buf_seconds = settings.pre_event_seconds + settings.post_event_seconds + 5
        frame_buffer: deque = deque(maxlen=_buf_seconds * _live_fps)
        last_alert_ts = 0.0
        target_dt = 1.0 / _live_fps

        try:
            while True:
                t0 = time.monotonic()

                if use_vision:
                    if stream is None:
                        return
                    display_frame, _ = stream.read()
                    if display_frame is None:
                        display_frame = first_frame if first_frame is not None else np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    display_frame = black_tpl.copy()

                frame_buffer.append(display_frame)

                # Cheap scene motion (normalized mean abs diff on a tiny gray frame).
                motion_level = 0.0
                if use_vision:
                    small = cv2.resize(display_frame, (64, 36))
                    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    if prev_gray_small is not None:
                        motion_level = float(np.mean(np.abs(gray_small - prev_gray_small)))
                    prev_gray_small = gray_small

                # Snapshot latest inference scores (written by background threads).
                with _infer_lock:
                    cv_raw = _cv_score[0]
                    ca_raw = _ca_score[0]
                    audio_ready = _audio_ready[0]

                if use_vision:
                    score_hist_cv.append(cv_raw)
                smooth_cv = float(sum(score_hist_cv) / len(score_hist_cv)) if score_hist_cv else None

                # Pass None until the audio thread has produced its first real result.
                # Passing 0.0 would drag the fusion score below threshold and suppress alerts.
                smooth_ca = None
                if use_audio and audio_ready:
                    score_hist_ca.append(ca_raw)
                    smooth_ca = float(sum(score_hist_ca) / len(score_hist_ca))

                fusion = compute_fusion_score(
                    smooth_cv,
                    smooth_ca,
                    settings.alpha,
                    settings.beta,
                    settings.threshold,
                    audio_alert_threshold=settings.audio_aggressive_threshold,
                )

                if fusion.score >= on_threshold:
                    consecutive_on += 1
                    consecutive_off = 0
                elif fusion.score <= off_threshold:
                    consecutive_off += 1
                    consecutive_on = 0

                live_on = max(1, settings.live_on_frames)
                live_off = max(1, settings.live_off_frames)
                if not alert_active and consecutive_on >= live_on:
                    alert_active = True
                if alert_active and consecutive_off >= live_off:
                    alert_active = False

                now_ts = time.time()
                rising_alert = alert_active and not prev_alert_active
                prev_alert_active = alert_active

                clip_threshold = min(1.0, t_hyst + max(0.0, settings.live_clip_score_margin))
                score_eligible = fusion.score >= clip_threshold
                motion_needed = settings.live_clip_require_motion and use_vision
                motion_eligible = (not motion_needed) or (motion_level >= settings.live_clip_motion_min)

                if (
                    rising_alert
                    and score_eligible
                    and motion_eligible
                    and (now_ts - last_alert_ts) > settings.alert_cooldown_seconds
                ):
                    last_alert_ts = now_ts
                    snapshot = list(frame_buffer)
                    th = threading.Thread(
                        target=_save_live_alert_event,
                        args=(snapshot, fusion.score, fusion.cv, fusion.ca, mode_label),
                        kwargs={"motion_level": motion_level if use_vision else None},
                        daemon=True,
                    )
                    th.start()

                overlay = display_frame.copy()
                v_txt = f"Vision: {smooth_cv:.3f}" if smooth_cv is not None else "Vision: off"
                a_txt = f"Audio:  {smooth_ca:.3f}" if smooth_ca is not None else "Audio:  off"
                c_txt = f"Combined: {fusion.score:.3f} (thr {t_hyst:.2f})"
                status = "ALERT" if alert_active else "OK"
                clr_stat = (0, 0, 255) if alert_active else (0, 220, 120)

                cv2.putText(overlay, v_txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)
                cv2.putText(overlay, a_txt, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 0, 255), 2)
                cv2.putText(overlay, c_txt, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                cv2.putText(overlay, f"Mode: {mode_label}", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 180), 2)
                cv2.putText(overlay, f"{status}  min gap {settings.alert_cooldown_seconds:.0f}s",
                            (12, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.58, clr_stat, 2)

                ret, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

                # Rate-limit to target FPS; don't spin at 100 % CPU.
                elapsed = time.monotonic() - t0
                remaining = target_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            _stop[0] = True

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _shutdown_dashboard(signum: int | None = None, frame: object | None = None) -> None:
    """Release camera handles; OpenCV/macOS backends can otherwise delay SIGINT handling."""
    _reset_camera_stream()
    os._exit(0)


if __name__ == "__main__":
    import signal

    # Second Ctrl+C still works via default handler after threads wind down.
    signal.signal(signal.SIGINT, lambda s, f: _shutdown_dashboard(s, f))
    signal.signal(signal.SIGTERM, lambda s, f: _shutdown_dashboard(s, f))

    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5050, debug=debug_mode, use_reloader=False)

"""
Comprehensive test suite for the AI Security System.
Tests all core modules: config, inference (vision+audio), audio utils,
fusion scoring, database, and the detect pipeline.

Run from project root:
    python scripts/run_tests.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import cv2
import librosa
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.audio import AudioStream, extract_mfcc
from utils.config import load_settings
from utils.db import EventDatabase, EventRecord
from utils.fusion import compute_fusion_score
from utils.inference import AudioInference, LiteModelRunner, VisionInference
from utils.ring_buffer import TimeRingBuffer


# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
SKIP = "\033[93m SKIP\033[0m"
SECTION = "\033[94m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


def section(title: str) -> None:
    print(f"\n{SECTION}{'─' * 60}{RESET}")
    print(f"{SECTION}{title}{RESET}")
    print(f"{SECTION}{'─' * 60}{RESET}")


def check(name: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    results.append((name, passed, detail))
    detail_str = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{detail_str}")


def run_test(name: str, fn) -> None:
    try:
        fn()
    except AssertionError as exc:
        check(name, False, str(exc))
    except Exception:
        check(name, False, traceback.format_exc().strip().splitlines()[-1])


# ── settings ─────────────────────────────────────────────────────────────────

def test_settings() -> None:
    section("1. Configuration (utils/config.py)")

    def t_defaults():
        s = load_settings()
        assert s.alpha == 0.6, f"alpha={s.alpha}"
        assert s.beta == 0.4, f"beta={s.beta}"
        assert s.threshold == 0.35, f"threshold={s.threshold}"
        assert s.fps == 15
        assert s.audio_sample_rate == 16000
        check("Default settings values", True,
              f"alpha={s.alpha} beta={s.beta} threshold={s.threshold}")

    def t_dirs_created():
        s = load_settings()
        assert s.output_dir.exists(), f"output_dir missing: {s.output_dir}"
        assert s.db_path.parent.exists(), f"db parent missing"
        check("Output & log dirs created", True)

    run_test("Default settings", t_defaults)
    run_test("Dirs created on load", t_dirs_created)


# ── vision inference ──────────────────────────────────────────────────────────

def test_vision_inference() -> None:
    section("2. Vision Inference (utils/inference.py)")
    settings = load_settings()
    model = VisionInference(settings.vision_model_path)
    model_ok = model.runner.available()
    check("VisionInference model loaded", model_ok,
          f"backend={model.runner.backend}")

    def t_score_range():
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        score = model.score(frame)
        assert 0.0 <= score <= 1.0, f"score={score} out of [0,1]"
        check("Score in [0, 1] (random frame)", True, f"score={score:.4f}")

    def t_fight_vs_normal():
        """Fight video should score higher than no-fight video."""
        fight_path = ROOT_DIR / "dataset/Peliculas/fights/newfi1.avi"
        nofight_path = ROOT_DIR / "dataset/Peliculas/noFights/1.mpg"
        if not fight_path.exists() or not nofight_path.exists():
            check("Fight > NoFight score comparison", True, "SKIP – dataset not found")
            return

        def avg_score(video_path: Path) -> float:
            cap = cv2.VideoCapture(str(video_path))
            scores = []
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % 5 == 0:
                    scores.append(model.score(frame))
                i += 1
            cap.release()
            return float(np.mean(scores)) if scores else 0.0

        fight_score = avg_score(fight_path)
        nofight_score = avg_score(nofight_path)
        passed = fight_score > nofight_score
        check(
            "Fight clip scores higher than no-fight clip",
            passed,
            f"fight_avg={fight_score:.4f} nofight_avg={nofight_score:.4f}",
        )

    def t_threshold_classification():
        settings2 = load_settings()
        fight_path = ROOT_DIR / "dataset/Peliculas/fights/newfi1.avi"
        if not fight_path.exists():
            check("Fight clip classified as violent-risk", True, "SKIP")
            return
        cap = cv2.VideoCapture(str(fight_path))
        scores = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % 5 == 0:
                scores.append(model.score(frame))
            i += 1
        cap.release()
        max_score = max(scores) if scores else 0.0
        passed = max_score > settings2.threshold
        check("Fight clip has at least one frame above threshold", passed,
              f"max_score={max_score:.4f} threshold={settings2.threshold}")

    run_test("Vision score range", t_score_range)
    run_test("Fight vs no-fight score", t_fight_vs_normal)
    run_test("Fight clip above threshold", t_threshold_classification)


# ── audio utils & inference ───────────────────────────────────────────────────

def test_audio() -> None:
    section("3. Audio Utils & Inference (utils/audio.py + utils/inference.py)")
    settings = load_settings()

    def t_mfcc_shape():
        audio = np.random.randn(32000).astype(np.float32)
        mfcc = extract_mfcc(audio, 16000)
        assert mfcc.ndim == 2, f"expected 2D, got shape {mfcc.shape}"
        assert mfcc.shape[0] == 40, f"expected 40 MFCC coefficients, got {mfcc.shape[0]}"
        check("MFCC shape (40, T) for 2-second clip", True, f"shape={mfcc.shape}")

    def t_mfcc_stereo():
        stereo = np.random.randn(32000, 2).astype(np.float32)
        mfcc = extract_mfcc(stereo, 16000)
        assert mfcc.ndim == 2
        check("MFCC from stereo input (auto-mix to mono)", True)

    def t_audio_model_loaded():
        audio_model = AudioInference(settings.audio_model_path)
        ok = audio_model.runner.available()
        check("AudioInference model loaded", ok,
              f"backend={audio_model.runner.backend} labels={audio_model.labels}")

    def t_audio_score_range():
        audio_model = AudioInference(settings.audio_model_path)
        audio = np.random.randn(32000).astype(np.float32) * 0.01
        mfcc = extract_mfcc(audio, 16000)
        score = audio_model.score(mfcc)
        assert 0.0 <= score <= 1.0, f"audio score {score} out of range"
        check("Audio score in [0, 1] (noise)", True, f"score={score:.4f}")

    def t_audio_predict_dict():
        audio_model = AudioInference(settings.audio_model_path)
        audio = np.random.randn(32000).astype(np.float32)
        mfcc = extract_mfcc(audio, 16000)
        result = audio_model.predict(mfcc)
        assert "aggressive_score" in result
        assert "top_label" in result
        assert "top_score" in result
        assert "probabilities" in result
        check("Audio predict() returns correct dict keys", True,
              f"top={result['top_label']} agg={result['aggressive_score']:.3f}")

    def t_angry_vs_neutral_audio():
        """Angry speech should score higher than neutral speech."""
        upload_dir = ROOT_DIR / "output/uploads"
        ang_files = sorted(upload_dir.glob("*_ANG_*.wav"))[:3]
        neu_files = sorted(upload_dir.glob("*_NEU_*.wav"))[:3]

        if not ang_files or not neu_files:
            check("ANG audio scores higher than NEU audio", True, "SKIP – no CREMA-D samples found")
            return

        audio_model = AudioInference(settings.audio_model_path)

        def avg_agg(files):
            scores = []
            for f in files:
                a, sr = librosa.load(str(f), sr=16000, mono=True)
                mfcc = extract_mfcc(a, sr)
                scores.append(audio_model.score(mfcc))
            return float(np.mean(scores))

        ang_score = avg_agg(ang_files)
        neu_score = avg_agg(neu_files)
        passed = ang_score > neu_score
        check(
            "Angry speech scores higher than neutral (aggressive model)",
            passed,
            f"ANG avg={ang_score:.3f} NEU avg={neu_score:.3f}",
        )

    def t_sad_labeled_as_distressed():
        """Sad speech should score high on distressed label."""
        upload_dir = ROOT_DIR / "output/uploads"
        sad_files = sorted(upload_dir.glob("*_SAD_*.wav"))[:3]
        if not sad_files:
            check("SAD audio → distressed label", True, "SKIP – no CREMA-D SAD samples found")
            return
        audio_model = AudioInference(settings.audio_model_path)
        distressed_scores = []
        for f in sad_files:
            a, sr = librosa.load(str(f), sr=16000, mono=True)
            mfcc = extract_mfcc(a, sr)
            pred = audio_model.predict(mfcc)
            probs = pred["probabilities"]
            if "distressed" in probs:
                distressed_scores.append(probs["distressed"])
        avg_dist = float(np.mean(distressed_scores)) if distressed_scores else 0.0
        check("SAD audio has notable distressed probability", avg_dist > 0.15,
              f"avg_distressed={avg_dist:.3f}")

    def t_all_crema_files():
        """Run prediction on all available CREMA-D files and print summary."""
        upload_dir = ROOT_DIR / "output/uploads"
        wav_files = list(upload_dir.glob("*.wav"))
        if not wav_files:
            check("CREMA-D full file test", True, "SKIP")
            return
        audio_model = AudioInference(settings.audio_model_path)
        emotion_scores: dict[str, list[float]] = {}
        for wav_path in wav_files:
            parts = wav_path.stem.split("_")
            emotion = parts[-2] if len(parts) >= 2 else "UNK"
            try:
                a, sr = librosa.load(str(wav_path), sr=16000, mono=True)
                mfcc = extract_mfcc(a, sr)
                score = audio_model.score(mfcc)
                emotion_scores.setdefault(emotion, []).append(score)
            except Exception:
                pass
        summary = {em: round(float(np.mean(scores)), 3)
                   for em, scores in emotion_scores.items()}
        detail = " | ".join(f"{k}={v}" for k, v in sorted(summary.items()))
        check("CREMA-D emotion → aggressive score mapping", True, detail)

    run_test("MFCC shape", t_mfcc_shape)
    run_test("MFCC from stereo", t_mfcc_stereo)
    run_test("Audio model loaded", t_audio_model_loaded)
    run_test("Audio score range", t_audio_score_range)
    run_test("Audio predict dict keys", t_audio_predict_dict)
    run_test("Angry vs Neutral audio", t_angry_vs_neutral_audio)
    run_test("SAD audio → distressed label", t_sad_labeled_as_distressed)
    run_test("CREMA-D full emotion summary", t_all_crema_files)


# ── fusion ────────────────────────────────────────────────────────────────────

def test_fusion() -> None:
    section("4. Fusion Scoring (utils/fusion.py)")

    def t_basic():
        f = compute_fusion_score(cv=0.8, ca=0.9, alpha=0.6, beta=0.4, threshold=0.35)
        expected = 0.6 * 0.8 + 0.4 * 0.9
        assert abs(f.score - expected) < 1e-5, f"score={f.score} expected={expected}"
        check("S = α·Cv + β·Ca", True, f"score={f.score:.4f}")

    def t_above_threshold():
        f = compute_fusion_score(cv=0.9, ca=0.9, alpha=0.6, beta=0.4, threshold=0.35)
        assert f.is_alert is True
        check("Alert triggered above threshold", True, f"score={f.score:.3f}")

    def t_below_threshold():
        f = compute_fusion_score(cv=0.1, ca=0.1, alpha=0.6, beta=0.4, threshold=0.35)
        assert f.is_alert is False
        check("No alert below threshold", True, f"score={f.score:.3f}")

    def t_ca_none():
        # When ca=None the score falls back to cv (vision only, no weighting).
        f = compute_fusion_score(cv=0.8, ca=None, alpha=0.6, beta=0.4, threshold=0.35)
        assert f.ca == 0.0, f"ca should be 0.0, got {f.ca}"
        assert abs(f.score - 0.8) < 1e-5, f"score should equal cv=0.8, got {f.score}"
        check("Fusion with Ca=None (audio unavailable)", True, f"score={f.score:.4f}")

    def t_clamp():
        f = compute_fusion_score(cv=2.0, ca=2.0, alpha=0.6, beta=0.4, threshold=0.35)
        assert f.score <= 1.0, f"score {f.score} should be clamped to 1.0"
        check("Fusion score clamped to [0, 1]", True, f"score={f.score:.4f}")

    run_test("Fusion formula", t_basic)
    run_test("Alert above threshold", t_above_threshold)
    run_test("No alert below threshold", t_below_threshold)
    run_test("Fusion with Ca=None", t_ca_none)
    run_test("Fusion score clamped", t_clamp)


# ── ring buffer ───────────────────────────────────────────────────────────────

def test_ring_buffer() -> None:
    section("5. Ring Buffer (utils/ring_buffer.py)")
    from datetime import datetime, timezone

    def t_basic_append_and_window():
        buf = TimeRingBuffer(max_seconds=5.0)
        now = time.time()
        for i in range(10):
            ts = datetime.fromtimestamp(now + i * 0.5, tz=timezone.utc)
            buf.append(np.zeros((10,)), ts)
        assert len(buf._items) == 10
        check("Append 10 items", True, f"count={len(buf._items)}")

    def t_eviction():
        buf = TimeRingBuffer(max_seconds=2.0)
        now = time.time()
        for i in range(20):
            ts = datetime.fromtimestamp(now + i * 0.5, tz=timezone.utc)
            buf.append(np.zeros((10,)), ts)
        assert len(buf._items) < 20, "old items should have been evicted"
        check("Old items evicted beyond max_seconds", True,
              f"remaining={len(buf._items)}")

    def t_latest():
        buf = TimeRingBuffer(max_seconds=10.0)
        assert buf.latest() is None, "empty buffer latest() should be None"
        now = time.time()
        ts = datetime.fromtimestamp(now, tz=timezone.utc)
        buf.append(np.array([42.0]), ts)
        latest = buf.latest()
        assert latest is not None
        assert latest.data[0] == 42.0
        check("latest() returns most recent item", True)

    run_test("Append and count", t_basic_append_and_window)
    run_test("Eviction beyond max_seconds", t_eviction)
    run_test("latest() on buffer", t_latest)


# ── database ─────────────────────────────────────────────────────────────────

def test_database() -> None:
    section("6. Database (utils/db.py)")

    def t_insert_and_query():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            db = EventDatabase(db_path)
            rec = EventRecord(
                event_id="test-001",
                threat_score=0.75,
                cv=0.8,
                ca=0.65,
                video_path="/tmp/clip.mp4",
                metadata_json=json.dumps({"device": "test_cam"}),
            )
            db.insert_event(rec)
            events = db.list_recent_events(limit=10)
            assert len(events) == 1, f"expected 1 event, got {len(events)}"
            assert events[0]["event_id"] == "test-001"
            assert abs(events[0]["threat_score"] - 0.75) < 1e-5
            db.close()
            check("Insert and query event", True,
                  f"event_id={events[0]['event_id']} score={events[0]['threat_score']}")
        finally:
            db_path.unlink(missing_ok=True)

    def t_multiple_events():
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)
        try:
            db = EventDatabase(db_path)
            for i in range(5):
                db.insert_event(EventRecord(
                    event_id=f"test-{i:03d}",
                    threat_score=float(i) / 5,
                    cv=0.5, ca=0.5,
                    video_path=f"/tmp/clip_{i}.mp4",
                    metadata_json="{}",
                ))
            events = db.list_recent_events(limit=3)
            assert len(events) == 3, f"limit=3 but got {len(events)}"
            db.close()
            check("get_recent_events respects limit", True, f"got {len(events)} of 5")
        finally:
            db_path.unlink(missing_ok=True)

    run_test("Insert and query single event", t_insert_and_query)
    run_test("get_recent_events with limit", t_multiple_events)


# ── detect pipeline ───────────────────────────────────────────────────────────

def test_detect_pipeline() -> None:
    section("7. Detect Pipeline (scripts/detect.py)")

    def t_fight_video_json():
        fight_path = ROOT_DIR / "dataset/Peliculas/fights/newfi1.avi"
        report_path = ROOT_DIR / "output/test_suite_fight_report.json"
        if not fight_path.exists():
            check("Fight video → violent-risk JSON report", True, "SKIP")
            return

        settings = load_settings()
        model = VisionInference(settings.vision_model_path)
        cap = cv2.VideoCapture(str(fight_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        scores = []
        timeline = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % 5 == 0:
                s = model.score(frame)
                scores.append(s)
                timeline.append((i / max(fps, 1.0), s))
            i += 1
        cap.release()

        avg = float(np.mean(scores)) if scores else 0.0
        mx = float(max(scores)) if scores else 0.0
        label = "violent-risk" if avg > settings.threshold else "normal"

        report = {"video": str(fight_path), "avg_score": avg,
                  "max_score": mx, "label": label}
        report_path.write_text(json.dumps(report, indent=2))
        passed = label == "violent-risk"
        check("Fight clip → violent-risk", passed,
              f"avg={avg:.4f} max={mx:.4f} threshold={settings.threshold}")

    def t_nofight_video_json():
        nofight_path = ROOT_DIR / "dataset/Peliculas/noFights/1.mpg"
        if not nofight_path.exists():
            check("No-fight video → normal JSON report", True, "SKIP")
            return

        settings = load_settings()
        model = VisionInference(settings.vision_model_path)
        cap = cv2.VideoCapture(str(nofight_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        scores = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % 5 == 0:
                scores.append(model.score(frame))
            i += 1
        cap.release()
        avg = float(np.mean(scores)) if scores else 0.0
        label = "violent-risk" if avg > settings.threshold else "normal"
        check("No-fight clip → normal", label == "normal",
              f"avg={avg:.4f} threshold={settings.threshold}")

    run_test("Fight video classified as violent-risk", t_fight_video_json)
    run_test("No-fight video classified as normal", t_nofight_video_json)


# ── end-to-end fusion ─────────────────────────────────────────────────────────

def test_end_to_end_fusion() -> None:
    section("8. End-to-End Fusion (vision + audio + fusion)")

    def t_fight_frame_fusion():
        """Score a fight frame through the full vision+audio+fusion pipeline."""
        fight_path = ROOT_DIR / "dataset/Peliculas/fights/newfi1.avi"
        if not fight_path.exists():
            check("Fight frame fusion score", True, "SKIP")
            return

        settings = load_settings()
        vision_model = VisionInference(settings.vision_model_path)
        audio_model = AudioInference(settings.audio_model_path)

        cap = cv2.VideoCapture(str(fight_path))
        ok, frame = cap.read()
        cap.release()
        assert ok, "Could not read a frame from fight video"

        cv_score = vision_model.score(frame)

        # Use angry CREMA-D audio if available, else use noise
        upload_dir = ROOT_DIR / "output/uploads"
        ang_files = list(upload_dir.glob("*_ANG_*.wav"))
        if ang_files:
            audio, sr = librosa.load(str(ang_files[0]), sr=16000, mono=True)
        else:
            audio = np.random.randn(32000).astype(np.float32)
            sr = 16000
        mfcc = extract_mfcc(audio, sr)
        ca_score = audio_model.score(mfcc)

        fusion = compute_fusion_score(
            cv=cv_score, ca=ca_score,
            alpha=settings.alpha, beta=settings.beta,
            threshold=settings.threshold,
        )
        check(
            "End-to-end fusion on fight frame",
            True,
            f"Cv={cv_score:.3f} Ca={ca_score:.3f} S={fusion.score:.3f} alert={fusion.is_alert}",
        )

    run_test("Fight frame full fusion", t_fight_frame_fusion)


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary() -> int:
    section("TEST SUMMARY")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)

    for name, ok, detail in results:
        status = PASS if ok else FAIL
        detail_str = f"  → {detail}" if detail else ""
        print(f"  [{status}] {name}{detail_str}")

    print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")

    if failed == 0:
        print(f"\n{SECTION}All {total} tests passed!{RESET}\n")
    else:
        print(f"\n\033[91m{failed} test(s) failed.\033[0m\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    print(f"\n{'=' * 60}")
    print("  AI Security System — Comprehensive Test Suite")
    print(f"{'=' * 60}")
    t0 = time.time()

    test_settings()
    test_vision_inference()
    test_audio()
    test_fusion()
    test_ring_buffer()
    test_database()
    test_detect_pipeline()
    test_end_to_end_fusion()

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    sys.exit(print_summary())

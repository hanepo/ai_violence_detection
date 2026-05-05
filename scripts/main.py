from __future__ import annotations

from collections import deque
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow running this file from either project root or scripts/ directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.audio import AudioStream, extract_mfcc
from utils.buzzer import Buzzer
from utils.config import load_settings
from utils.db import EventDatabase, EventRecord
from utils.email_notifier import EmailNotifier
from utils.fusion import compute_fusion_score
from utils.inference import AudioInference, VisionInference
from utils.logging_utils import setup_logging
from utils.mqtt_notifier import MQTTNotifier
from utils.ring_buffer import TimeRingBuffer
from utils.video import VideoCaptureStream, write_playable_mp4

try:
	from utils.pir import PIRSensor
except Exception:
	PIRSensor = None


def _append_audio_chunk(audio_stream: AudioStream, audio_buffer: TimeRingBuffer) -> None:
	wav, wav_ts = audio_stream.read_chunk()
	audio_buffer.append(wav, wav_ts)


def _collect_post_event_context(
	video_stream: VideoCaptureStream,
	audio_stream: AudioStream,
	video_buffer: TimeRingBuffer,
	audio_buffer: TimeRingBuffer,
	settings,
) -> None:
	deadline = time.time() + max(0.0, float(settings.post_event_seconds))
	while time.time() < deadline:
		frame, frame_ts = video_stream.read()
		if frame is not None:
			video_buffer.append(frame, frame_ts)
		_append_audio_chunk(audio_stream, audio_buffer)
		time.sleep(max(0.001, 1.0 / settings.fps))


def _persist_alert_event(
	db: EventDatabase,
	mqtt: MQTTNotifier,
	email_notifier: EmailNotifier,
	buzzer: Buzzer,
	settings,
	video_buffer: TimeRingBuffer,
	center: datetime,
	fusion,
) -> None:
	event_id = str(uuid.uuid4())

	window = video_buffer.window(
		center=center,
		before_seconds=settings.pre_event_seconds,
		after_seconds=settings.post_event_seconds,
	)
	frames = [item.data for item in window]
	clip_name = f"event_{event_id}.mp4"
	clip_path = settings.output_dir / clip_name
	write_playable_mp4(frames, clip_path, settings.fps)
	clip_duration_seconds = float(len(frames) / max(settings.fps, 1))

	metadata = {
		"device_id": settings.device_id,
		"camera_name": settings.camera_name,
		"mode": (
			"always_on"
			if not settings.enable_pir
			else ("mock_pir" if settings.use_mock_pir else "gpio_pir")
		),
		"pir_pin": settings.pir_pin if settings.enable_pir else None,
		"buzzer_enabled": settings.enable_buzzer,
		"video_frames": len(frames),
		"alert_timestamp_utc": center.isoformat(),
		"clip_duration_seconds": round(clip_duration_seconds, 3),
		"pre_event_seconds": settings.pre_event_seconds,
		"post_event_seconds": settings.post_event_seconds,
	}

	db.insert_event(
		EventRecord(
			event_id=event_id,
			threat_score=fusion.score,
			cv=fusion.cv,
			ca=fusion.ca,
			video_path=str(clip_path),
			metadata_json=json.dumps(metadata),
		)
	)

	payload = MQTTNotifier.build_payload(
		device_id=settings.device_id,
		threat_score=fusion.score,
		event_id=event_id,
	)
	if mqtt.publish_alert(payload):
		logging.warning("ALERT SENT: %s", payload)
	else:
		logging.warning("ALERT LOGGED but MQTT broker unavailable: %s", payload)

	if email_notifier.enabled:
		email_ok = email_notifier.send_alert(
			event_id=event_id,
			threat_score=fusion.score,
			camera_name=settings.camera_name,
			alert_timestamp_utc=center.isoformat(),
			clip_path=clip_path,
			clip_duration_seconds=clip_duration_seconds,
		)
		if email_ok:
			logging.warning("EMAIL ALERT SENT: event=%s recipients=%s", event_id, ",".join(email_notifier.recipients))
		else:
			logging.warning("EMAIL ALERT FAILED: event=%s (check SMTP env config)", event_id)

	buzzer.beep_pattern(
		repeats=settings.buzzer_repeats,
		on_seconds=settings.buzzer_beep_seconds,
		off_seconds=settings.buzzer_pause_seconds,
	)


def run() -> None:
	settings = load_settings()
	setup_logging(Path("logs/system.log"))

	logging.info("Initializing system")
	db = EventDatabase(settings.db_path)
	mqtt = MQTTNotifier(settings.mqtt_broker, settings.mqtt_port, settings.mqtt_topic)
	email_notifier = EmailNotifier(
		enabled=settings.enable_email_alerts,
		smtp_host=settings.smtp_host,
		smtp_port=settings.smtp_port,
		smtp_username=settings.smtp_username,
		smtp_password=settings.smtp_password,
		sender=settings.smtp_sender,
		recipients_csv=settings.smtp_recipients,
		use_tls=settings.smtp_use_tls,
		use_ssl=settings.smtp_use_ssl,
		subject_prefix=settings.email_subject_prefix,
		dashboard_base_url=settings.dashboard_base_url,
	)
	pir = None
	if settings.enable_pir:
		if PIRSensor is None:
			raise RuntimeError("PIR is enabled but utils.pir could not be imported")
		pir = PIRSensor(pin=settings.pir_pin, use_mock=settings.use_mock_pir)
	buzzer = Buzzer(pin=settings.buzzer_pin, enabled=settings.enable_buzzer, use_mock=settings.use_mock_pir)

	video_stream = VideoCaptureStream(
		camera_index=settings.camera_index,
		width=settings.frame_width,
		height=settings.frame_height,
		fps=settings.fps,
	)

	audio_stream = AudioStream(
		sample_rate=settings.audio_sample_rate,
		channels=settings.audio_channels,
		chunk_seconds=settings.audio_chunk_seconds,
	)

	vision = VisionInference(settings.vision_model_path)
	audio = AudioInference(settings.audio_model_path)

	# Keep enough history to support pre + post window extraction.
	video_buffer = TimeRingBuffer(max_seconds=settings.pre_event_seconds + settings.post_event_seconds + 5)
	audio_buffer = TimeRingBuffer(max_seconds=settings.pre_event_seconds + settings.post_event_seconds + 5)

	if settings.enable_pir:
		logging.info("Entering standby state (PIR enabled on pin %d)", settings.pir_pin)
	else:
		logging.info("Entering always-on state (PIR disabled)")
	last_alert_ts = 0.0

	score_hist: deque[float] = deque(maxlen=max(1, settings.live_score_window))
	alert_active = False
	consecutive_on = 0
	consecutive_off = 0

	on_threshold = min(1.0, settings.threshold + settings.live_hysteresis)
	off_threshold = max(0.0, settings.threshold - settings.live_hysteresis)

	try:
		while True:
			if settings.enable_pir:
				if pir is None or not pir.motion_detected():
					time.sleep(0.2)
					continue

				logging.info("Motion detected on PIR pin %d, entering active state", settings.pir_pin)
				active_until = time.time() + settings.active_window_seconds
			else:
				# In always-on mode process continuously without standby gating.
				active_until = time.time() + max(0.05, 1.0 / max(settings.fps, 1))

			while time.time() < active_until:
				frame, frame_ts = video_stream.read()
				if frame is None:
					time.sleep(0.02)
					continue
				video_buffer.append(frame, frame_ts)

				raw_cv = vision.score(frame)
				score_hist.append(raw_cv)
				cv = float(sum(score_hist) / len(score_hist))

				_append_audio_chunk(audio_stream, audio_buffer)
				latest = audio_buffer.latest()
				if latest is not None:
					mfcc = extract_mfcc(latest.data, settings.audio_sample_rate)
					ca = audio.score(mfcc)
				else:
					ca = None

				fusion = compute_fusion_score(
					cv=cv,
					ca=ca,
					alpha=settings.alpha,
					beta=settings.beta,
					threshold=settings.threshold,
				)

				if fusion.score >= on_threshold:
					consecutive_on += 1
					consecutive_off = 0
				elif fusion.score <= off_threshold:
					consecutive_off += 1
					consecutive_on = 0

				if not alert_active and consecutive_on >= settings.live_on_frames:
					alert_active = True
				if alert_active and consecutive_off >= settings.live_off_frames:
					alert_active = False

				logging.info(
					"Fusion=%.3f Cv(raw=%.3f smooth=%.3f) Ca=%.3f StableAlert=%s",
					fusion.score,
					raw_cv,
					fusion.cv,
					fusion.ca,
					alert_active,
				)

				if alert_active and (time.time() - last_alert_ts) > settings.alert_cooldown_seconds:
					last_alert_ts = time.time()
					center = datetime.now(timezone.utc)

					# Ensure we actually have post-event context before extracting clip.
					_collect_post_event_context(
						video_stream=video_stream,
						audio_stream=audio_stream,
						video_buffer=video_buffer,
						audio_buffer=audio_buffer,
						settings=settings,
					)

					_persist_alert_event(
						db=db,
						mqtt=mqtt,
						email_notifier=email_notifier,
						buzzer=buzzer,
						settings=settings,
						video_buffer=video_buffer,
						center=center,
						fusion=fusion,
					)

				time.sleep(max(0.001, 1.0 / settings.fps))

			if settings.enable_pir:
				logging.info("Returning to standby state")

	except KeyboardInterrupt:
		logging.info("Stopping system")
	finally:
		video_stream.release()
		db.close()
		mqtt.close()
		buzzer.cleanup()
		if pir is not None:
			pir.cleanup()


if __name__ == "__main__":
	run()

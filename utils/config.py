from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    device_id: str = os.getenv("DEVICE_ID", "corridor_a_cam01")
    camera_name: str = os.getenv("CAMERA_NAME", os.getenv("DEVICE_ID", "corridor_a_cam01"))
    mqtt_broker: str = os.getenv("MQTT_BROKER", "127.0.0.1")
    mqtt_port: int = int(os.getenv("MQTT_PORT", "1883"))
    mqtt_topic: str = os.getenv("MQTT_TOPIC", "school/corridor_a/alerts")

    # Optional SMTP email alerts.
    enable_email_alerts: bool = os.getenv("ENABLE_EMAIL_ALERTS", "0") == "1"
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    smtp_sender: str = os.getenv("SMTP_SENDER", "")
    smtp_recipients: str = os.getenv("SMTP_RECIPIENTS", "")
    smtp_use_tls: bool = os.getenv("SMTP_USE_TLS", "1") == "1"
    smtp_use_ssl: bool = os.getenv("SMTP_USE_SSL", "0") == "1"
    email_subject_prefix: str = os.getenv("EMAIL_SUBJECT_PREFIX", "[AI Security Alert]")
    dashboard_base_url: str = os.getenv("DASHBOARD_BASE_URL", "")

    db_path: Path = Path(os.getenv("DB_PATH", "logs/events.db"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "output"))

    # Fusion: S = alpha * Cv + beta * Ca
    alpha: float = float(os.getenv("FUSION_ALPHA", "0.6"))
    beta: float = float(os.getenv("FUSION_BETA", "0.4"))
    threshold: float = float(os.getenv("FUSION_THRESHOLD", "0.35"))

    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    fps: int = int(os.getenv("VIDEO_FPS", "15"))
    frame_width: int = int(os.getenv("VIDEO_WIDTH", "640"))
    frame_height: int = int(os.getenv("VIDEO_HEIGHT", "480"))

    pir_pin: int = int(os.getenv("PIR_PIN", "17"))
    enable_pir: bool = os.getenv("ENABLE_PIR", "0") == "1"
    active_window_seconds: float = float(os.getenv("ACTIVE_WINDOW_SECONDS", "10.0"))
    alert_cooldown_seconds: float = float(os.getenv("ALERT_COOLDOWN_SECONDS", "8.0"))
    live_score_window: int = int(os.getenv("LIVE_SCORE_WINDOW", "6"))
    live_on_frames: int = int(os.getenv("LIVE_ON_FRAMES", "3"))
    live_off_frames: int = int(os.getenv("LIVE_OFF_FRAMES", "5"))
    live_hysteresis: float = float(os.getenv("LIVE_HYSTERESIS", "0.05"))

    audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    audio_channels: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    audio_chunk_seconds: float = float(os.getenv("AUDIO_CHUNK_SECONDS", "2.0"))
    audio_aggressive_threshold: float = float(os.getenv("AUDIO_AGGRESSIVE_THRESHOLD", "0.18"))

    # Dashboard mic probe only (status text), not used for threat fusion.
    mic_probe_active_rms: float = float(os.getenv("MIC_PROBE_ACTIVE_RMS", "0.004"))
    mic_probe_active_peak: float = float(os.getenv("MIC_PROBE_ACTIVE_PEAK", "0.035"))

    pre_event_seconds: int = int(os.getenv("PRE_EVENT_SECONDS", "10"))
    post_event_seconds: int = int(os.getenv("POST_EVENT_SECONDS", "10"))

    # Models are optional in early bring-up.
    vision_model_path: str = os.getenv("VISION_MODEL_PATH", "models/vision.tflite")
    audio_model_path: str = os.getenv("AUDIO_MODEL_PATH", "models/audio.tflite")

    # Upload test stabilization controls.
    upload_motion_p90_min: float = float(os.getenv("UPLOAD_MOTION_P90_MIN", "0.030"))
    upload_min_alert_seconds: float = float(os.getenv("UPLOAD_MIN_ALERT_SECONDS", "0.50"))
    upload_risk_ratio_min: float = float(os.getenv("UPLOAD_RISK_RATIO_MIN", "0.10"))
    upload_strong_peak_delta: float = float(os.getenv("UPLOAD_STRONG_PEAK_DELTA", "0.18"))

    # Optional buzzer alarm on confirmed alerts (Raspberry Pi).
    enable_buzzer: bool = os.getenv("ENABLE_BUZZER", "0") == "1"
    buzzer_pin: int = int(os.getenv("BUZZER_PIN", "18"))
    buzzer_beep_seconds: float = float(os.getenv("BUZZER_BEEP_SECONDS", "0.15"))
    buzzer_pause_seconds: float = float(os.getenv("BUZZER_PAUSE_SECONDS", "0.08"))
    buzzer_repeats: int = int(os.getenv("BUZZER_REPEATS", "3"))

    # Development aids for Mac-first testing.
    use_mock_pir: bool = os.getenv("USE_MOCK_PIR", "1") == "1"


def load_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

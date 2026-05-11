# AI Security System (Mac-first, Raspberry Pi deployment)

This project implements a multimodal edge AI pipeline aligned with your report:

- Vision + audio inference
- Fusion score: `S = alpha * Cv + beta * Ca`
- Always-on monitoring by default (optional PIR trigger mode)
- Local event logging + evidence clip saving
- MQTT network alerts

Development path:

1. Train and test on MacBook first.
2. Deploy the same pipeline to Raspberry Pi 5 (8GB).

Client deployment tutorial:

- See `CLIENT_HANDOVER_GUIDE.md` for complete end-user setup and operation steps.

## Project Structure

- `dataset/`: training data
- `models/`: trained/exported models (`.tflite`)
- `scripts/`: runtime + train + evaluate scripts
- `utils/`: shared modules (fusion, DB, MQTT, I/O, inference)
- `logs/`: runtime logs and SQLite DB
- `output/`: saved clips and artifacts

## 1) Local Setup (macOS/Linux)

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1.1) Local Setup (Windows PowerShell)

From project root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Important:

- Do not use `source .venv/bin/activate` on Windows PowerShell.
- For Windows CMD, use `.venv\Scripts\activate.bat`.

## 2) Run the System (Quick Start)

For most users, run the web dashboard only:

```bash
python scripts/dashboard.py
```

Open in browser:

```text
http://127.0.0.1:5050
```

For background mode:

```bash
fuser -k 5050/tcp 2>/dev/null || true
FLASK_DEBUG=0 nohup python scripts/dashboard.py >/tmp/dashboard.log 2>&1 < /dev/null &
```

To stop dashboard:

```bash
fuser -k 5050/tcp 2>/dev/null || true
```

To view dashboard logs:

```bash
tail -n 50 /tmp/dashboard.log
```

## 2.1) Generate Client Metrics (Vision)

If client asks for model curves/metrics artifacts, generate from existing threshold sweep data:

```bash
python scripts/generate_vision_metrics_report.py \
   --sweep-csv output/vision_threshold_sweep.csv \
   --threshold 0.35 \
   --out-dir output/model_metrics/vision
```

Output files include:

- `model_validation.json`
- `vision_threshold_metrics.csv`
- `confusion_matrix.png`
- `precision_recall_curve.png`
- `f1_confidence_curve.png`
- `precision_confidence_curve.png`
- `recall_confidence_curve.png`

Note: training loss/accuracy graphs require saving training history during training runs.
Current training scripts now save:

- `models/vision_training_history.csv`
- `models/vision_training_history.png`
- `models/audio_training_history.csv`
- `models/audio_training_history.png`

## 3) Optional Full Pipeline Mode

If you want fusion alerts, event clips, and MQTT alerts, run:

```bash
python scripts/main.py
```

Notes:

- PIR is disabled by default (`ENABLE_PIR=0`), so capture runs continuously.
- If you want PIR gating, set `ENABLE_PIR=1` and optionally `USE_MOCK_PIR=1` for development.
- Do not run `main.py` and `dashboard.py` on the same camera at the same time.
- Recorded alert clips are saved and shown in dashboard `Recent Alerts -> Replay`.

## 4) Raspberry Pi 5 Deployment

1. Install Raspberry Pi OS (64-bit), enable camera + SSH.
2. Connect hardware:
   - Camera Module 3 via CSI
   - USB microphone
   - Optional PIR (HC-SR501) to GPIO17
   - Ethernet to school LAN
3. Copy project and models to Pi.
4. Install dependencies in Pi venv.
5. Set runtime env vars:

```bash
sudo apt install -y python3-libcamera python3-picamera2 libcamera-tools libcamera-v4l2 libcap-dev

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.pi.txt
```

Notes:

- For Python 3.13 on recent Pi OS, this project uses `ai-edge-litert` instead of `tflite-runtime`.
- CSI cameras on Pi 5 use libcamera/Picamera2. If dashboard stream is black, verify system stack first:

```bash
rpicam-hello --list-cameras
```

```bash
export ENABLE_PIR=0
export MQTT_BROKER=<your_broker_ip>
export MQTT_TOPIC=school/corridor_a/alerts
export FUSION_THRESHOLD=0.35
export AUDIO_AGGRESSIVE_THRESHOLD=0.18

# Only if you want PIR-gated mode:
# export ENABLE_PIR=1
# export USE_MOCK_PIR=0
# export PIR_PIN=17

# Live dashboard / clip recording (reduce false positives; see scripts/dashboard.py)
export LIVE_SCORE_WINDOW=6
export LIVE_ON_FRAMES=12
export LIVE_OFF_FRAMES=8
export LIVE_HYSTERESIS=0.05
export LIVE_ALERT_ENTER_MARGIN=0.10
export LIVE_ALERT_EXIT_MARGIN=0.06
export LIVE_CLIP_SCORE_MARGIN=0.03
export LIVE_CLIP_REQUIRE_MOTION=1
export LIVE_CLIP_MOTION_MIN=0.014
export ALERT_COOLDOWN_SECONDS=8.0

# Upload test sensitivity tuning (reduce misses on difficult violence clips)
export UPLOAD_MOTION_P90_MIN=0.020
export UPLOAD_MIN_ALERT_SECONDS=0.35
export UPLOAD_RISK_RATIO_MIN=0.06
export UPLOAD_STRONG_PEAK_DELTA=0.12

# Clip length control (total clip = PRE_EVENT_SECONDS + POST_EVENT_SECONDS).
# Use 10+10=20 seconds (minimum) up to 20+20=40 seconds.
export PRE_EVENT_SECONDS=10
export POST_EVENT_SECONDS=10

# Optional SMTP email alerts to teacher.
export ENABLE_EMAIL_ALERTS=1
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_USERNAME=your_dummy_email@gmail.com
export SMTP_PASSWORD=your_app_password
export SMTP_SENDER=your_dummy_email@gmail.com
export SMTP_RECIPIENTS=teacher1@school.edu,teacher2@school.edu
export SMTP_USE_TLS=1
export SMTP_USE_SSL=0
export EMAIL_SUBJECT_PREFIX="[School AI Alert]"
export DASHBOARD_BASE_URL=http://192.168.1.24:5050

# Optional buzzer alarm
export ENABLE_BUZZER=1
export BUZZER_PIN=18
export BUZZER_REPEATS=3
python scripts/main.py
```

## 5) Security Notes

- Current baseline logs to SQLite (`logs/events.db`).
- For production, migrate to SQLCipher-backed DB file on Pi.
- Keep MQTT topic on private LAN and authenticated broker.
- Avoid transmitting raw audio/video over MQTT.

## 6) Next Enhancements

- Replace baseline vision runner with YOLOv8n exported to TFLite/ONNX.
- Replace baseline audio CNN with tuned MobileNet1D/2D MFCC model.
- Add Flask dashboard for status, logs, and sensitivity control.
- Add systemd service for 24/7 boot-time runtime.

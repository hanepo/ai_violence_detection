# AI Security System - Client Handover Guide

This guide is for the client who receives the Raspberry Pi device.
It explains what to install, how to connect Wi-Fi, how to run the dashboard, and how to troubleshoot common issues.

## 1) What the Client Receives

- Raspberry Pi 5 with Raspberry Pi OS installed
- Project folder: `~/ai-security-system`
- Camera connected (CSI)
- Optional USB microphone
- Optional PIR sensor (not required for live view)

## 2) Quick Start (Most Important)

If the goal is only live camera view in browser (24/7 preview), run dashboard only.

Do this on Raspberry Pi terminal:

```bash
cd ~/ai-security-system
source .venv/bin/activate
fuser -k 5050/tcp 2>/dev/null || true
FLASK_DEBUG=0 nohup python scripts/dashboard.py >/tmp/dashboard.log 2>&1 < /dev/null &
```

Then open from a phone/laptop on same network:

- `http://raspberrypi.local:5050` (or `http://aisecuritysystem.local:5050` if hostname was changed)
- If that does not work, use Pi IP: `http://<PI_IP>:5050`

## 3) Does Client Need to Change IP in Code?

No.

- IP address is assigned by the current network (DHCP).
- When client changes Wi-Fi, Pi gets a new IP automatically.
- No code update is needed for IP changes.

## 4) How Client Changes Wi-Fi

### Option A (Easiest): Monitor + Keyboard on Pi

1. Connect monitor, keyboard, and power.
2. Click Wi-Fi icon on Raspberry Pi desktop.
3. Select client Wi-Fi and enter password.
4. Confirm internet works.

### Option B: Headless (No Monitor)

Use Ethernet or any reachable method to SSH into Pi, then configure Wi-Fi with system tools.
If client is not technical, Option A is strongly recommended.

## 5) Find Pi IP Address

On Raspberry Pi terminal:

```bash
hostname -I
```

This prints one or more addresses. Use the local LAN IPv4 address (example: `192.168.1.24`).

## 6) Install Dependencies (If Reinstall Needed)

Only needed if environment was reset or new SD card was prepared.

```bash
sudo apt update
sudo apt install -y python3-venv python3-libcamera python3-picamera2 libcamera-tools libcamera-v4l2 libcap-dev

cd ~/ai-security-system
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.pi.txt
```

## 7) Run Modes

### Mode A: Live Camera Website Only (Recommended for Client)

Run only:

```bash
python scripts/dashboard.py
```

or background mode:

```bash
fuser -k 5050/tcp 2>/dev/null || true
FLASK_DEBUG=0 nohup python scripts/dashboard.py >/tmp/dashboard.log 2>&1 < /dev/null &
```

### Mode B: Full AI Alert Pipeline

Run:

```bash
python scripts/main.py
```

This does fusion alert logic, event logging, clip saving, and MQTT alerts.

Important:

- Do not run `main.py` and `dashboard.py` on the same camera at the same time unless designed for that setup.
- For stable live view, keep only `dashboard.py` running.

## 8) Common Problems and Fixes

### Problem: Port 5050 already in use

Error:

- "Address already in use"
- "Port 5050 is in use"

Fix:

```bash
fuser -k 5050/tcp 2>/dev/null || true
```

Then start dashboard again.

### Problem: Black camera on web page

Checks:

```bash
rpicam-hello --list-cameras
curl -s --max-time 5 http://127.0.0.1:5050/camera-feed -o /tmp/camfeed.bin || true
wc -c /tmp/camfeed.bin
```

- If byte count is large (not 0), stream is running.
- Refresh browser with hard reload.
- Ensure only one camera-consuming process is active.

### Problem: Colors look wrong (blue-ish skin)

Try swapping channels using environment flag:

```bash
fuser -k 5050/tcp 2>/dev/null || true
PICAM_SWAP_RB=1 FLASK_DEBUG=0 nohup python scripts/dashboard.py >/tmp/dashboard.log 2>&1 < /dev/null &
```

If color becomes worse, remove `PICAM_SWAP_RB=1` and restart.

### Problem: Cannot access by hostname

If `raspberrypi.local` does not open, use direct IP from `hostname -I`.

## 9) Basic Service Commands for Client

Start dashboard:

```bash
cd ~/ai-security-system
source .venv/bin/activate
fuser -k 5050/tcp 2>/dev/null || true
FLASK_DEBUG=0 nohup python scripts/dashboard.py >/tmp/dashboard.log 2>&1 < /dev/null &
```

Stop dashboard:

```bash
fuser -k 5050/tcp 2>/dev/null || true
```

See logs:

```bash
tail -n 50 /tmp/dashboard.log
```

## 10) Recommended Security Before Delivery

- Change default Pi password.
- Keep SSH enabled only if needed.
- Use strong Wi-Fi password.
- Keep MQTT broker private/authenticated if alerts are enabled.

## 11) Delivery Checklist

- [ ] Pi boots normally.
- [ ] Camera visible in dashboard.
- [ ] Client can open dashboard from another device on same Wi-Fi.
- [ ] Client knows how to reconnect Wi-Fi.
- [ ] Client has this guide.
- [ ] Client knows no code IP changes are required.

## 12) One-Line Answer for Client

"When Wi-Fi changes, the Raspberry Pi IP changes automatically. Just reconnect Pi to the new Wi-Fi and open the dashboard using the new IP or raspberrypi.local. No source code change is needed."

## 13) Zero-Support Delivery Mode (Recommended)

If you will not be there to guide the client, set auto-start before delivery.
Then the client only needs power + Wi-Fi + browser.

### Step A: Configure dashboard auto-start once (done by installer before delivery)

Run on Raspberry Pi:

```bash
cat <<'EOF' | sudo tee /etc/systemd/system/ai-dashboard.service >/dev/null
[Unit]
Description=AI Security Dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=aisecuritysystem
WorkingDirectory=/home/aisecuritysystem/ai-security-system
Environment=FLASK_DEBUG=0
ExecStart=/home/aisecuritysystem/ai-security-system/.venv/bin/python scripts/dashboard.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ai-dashboard.service
sudo systemctl restart ai-dashboard.service
sudo systemctl status ai-dashboard.service --no-pager
```

### Step B: Test reboot behavior once (done by installer before delivery)

```bash
sudo reboot
```

After reboot:

```bash
sudo systemctl status ai-dashboard.service --no-pager
```

### Step C: Give this simple instruction to client

1. Power on Raspberry Pi.
2. Connect Pi to your Wi-Fi (monitor + keyboard, first time only).
3. Open browser on same Wi-Fi.
4. Try `http://raspberrypi.local:5050`.
5. If not opening, find Pi IP with `hostname -I` on Pi and open `http://<PI_IP>:5050`.

No terminal commands are required from client after auto-start is configured.

## 14) Emergency Recovery Card (Give to Client)

If page does not open, run these exact commands on Pi terminal:

```bash
sudo systemctl restart ai-dashboard.service
sudo systemctl status ai-dashboard.service --no-pager
hostname -I
```

If service fails, run:

```bash
sudo journalctl -u ai-dashboard.service -n 80 --no-pager
```

Send that output to support.

## 15) Student Notes: Metrics and Tuning (Simple)

This section explains the model-evaluation items usually requested by supervisors/clients.

### A) Metrics Checklist and Where to Find Files

After generating metrics, check these files under `output/model_metrics/vision/`:

- `model_validation.json` -> selected threshold metrics summary
- `vision_threshold_metrics.csv` -> full threshold table
- `confusion_matrix.png` -> confusion matrix
- `precision_recall_curve.png` -> precision-recall curve
- `f1_confidence_curve.png` -> F1-confidence curve
- `precision_confidence_curve.png` -> precision-confidence curve
- `recall_confidence_curve.png` -> recall-confidence curve

Run command (from project root):

```bash
python scripts/generate_vision_metrics_report.py \
	--sweep-csv output/vision_threshold_sweep.csv \
	--threshold 0.35 \
	--out-dir output/model_metrics/vision
```

### B) Training Metrics and Loss Graphs

After retraining, these files are saved automatically:

- `models/vision_training_history.csv`
- `models/vision_training_history.png`
- `models/audio_training_history.csv`
- `models/audio_training_history.png`

### C) If Violence Video Is Not Detected (Tuning)

Use these environment variables to make upload detection more sensitive:

```bash
export UPLOAD_MOTION_P90_MIN=0.020
export UPLOAD_MIN_ALERT_SECONDS=0.35
export UPLOAD_RISK_RATIO_MIN=0.06
export UPLOAD_STRONG_PEAK_DELTA=0.12
```

Then restart dashboard and test again.

### D) Quick Explanation for Presentation

- The current vision pipeline is frame-level classification (not object bounding boxes).
- Final alert decision is based on score threshold and temporal smoothing/hysteresis.
- For stricter detection detail (person-level box), a detection model (for example YOLO) is required.

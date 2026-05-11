from __future__ import annotations

from datetime import datetime, timezone
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import sys

import cv2
import numpy as np


def _ffmpeg_binary() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None


def transcode_to_browser_mp4(src: Path, dest: Path, *, timeout: int = 300) -> bool:
    """Remux / re-encode to H.264 + yuv420p + faststart so Chrome/Safari can play the file."""
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        return False
    if not src.exists() or src.stat().st_size == 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    # -an: clip may have no audio; avoids rare demux issues in Chrome/Safari.
    # baseline + yuv420p + faststart: broad <video> compatibility.
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False


def write_playable_mp4(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    """Write an MP4 from BGR frames; prefer browser-playable H.264 when ffmpeg is available.

    OpenCV's default ``mp4v`` (MPEG-4 Part 2) is often not decodable in web browsers,
    so we always try to transcode to H.264 after a provisional write.
    """
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", dir=str(out_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_video(frames, tmp_path, fps)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            return
        if transcode_to_browser_mp4(tmp_path, out_path):
            tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.replace(out_path)
    except Exception:
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


def _import_picamera2():
    try:
        from picamera2 import Picamera2

        return Picamera2
    except Exception:
        # On Raspberry Pi, apt-installed camera modules live in dist-packages.
        dist_packages = "/usr/lib/python3/dist-packages"
        if dist_packages not in sys.path:
            sys.path.append(dist_packages)
        try:
            from picamera2 import Picamera2

            return Picamera2
        except Exception:
            return None


class VideoCaptureStream:
    """Thread-safe camera stream that captures frames in a dedicated background thread.

    macOS AVFoundation (and many camera backends) must be read on a consistent
    thread. Running cap.read() directly inside a Flask request thread causes
    intermittent failures. This class owns a capture thread and exposes the
    latest frame via a lock-protected variable so any calling thread can read it.
    """

    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        self._picam_swap_rb = os.getenv("PICAM_SWAP_RB", "0") == "1"
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: datetime = datetime.now(timezone.utc)
        self._stopped = False
        self._thread: threading.Thread | None = None
        self.picam2 = None
        self.cap = None

        # Try OpenCV first; start background capture thread on success.
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if cap.isOpened():
            self.cap = cap
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            # Wait up to 2 s for the first real frame.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                with self._lock:
                    if self._latest_frame is not None:
                        break
                time.sleep(0.05)
            return
        cap.release()

        # Fallback for Raspberry Pi CSI cameras via libcamera/Picamera2.
        Picamera2 = _import_picamera2()
        if Picamera2 is not None:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (width, height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    with self._lock:
                        if self._latest_frame is not None:
                            break
                    time.sleep(0.05)
            except Exception:
                if self.picam2 is not None:
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                self.picam2 = None

    def _capture_loop(self) -> None:
        while not self._stopped:
            frame = None
            ts = datetime.now(timezone.utc)

            if self.cap is not None:
                ok, f = self.cap.read()
                if ok and f is not None:
                    frame = f
            elif self.picam2 is not None:
                try:
                    f = self.picam2.capture_array()
                    if f is not None:
                        if len(f.shape) == 3 and f.shape[2] == 4:
                            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR if not self._picam_swap_rb else cv2.COLOR_RGBA2BGR)
                        elif len(f.shape) == 3 and f.shape[2] == 3 and self._picam_swap_rb:
                            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                        frame = f
                except Exception:
                    pass

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = ts
                # cap.read() blocks until the next frame on most backends, so no
                # artificial sleep needed; add a tiny yield to stay cooperative.
                time.sleep(0.001)
            else:
                time.sleep(0.02)

    def read(self) -> tuple[np.ndarray | None, datetime]:
        with self._lock:
            return (
                self._latest_frame.copy() if self._latest_frame is not None else None,
                self._latest_ts,
            )

    def release(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None


class BlackFrameVideoStream:
    """Placeholder capture when vision inference is disabled but loops expect frames."""

    def __init__(self, width: int, height: int):
        self._size = (height, width, 3)

    def read(self) -> tuple[np.ndarray, datetime]:
        return np.zeros(self._size, dtype=np.uint8), datetime.now(timezone.utc)

    def release(self) -> None:
        pass


def write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(max(fps, 1)), (w, h))
    if not writer.isOpened():
        writer.release()
        return
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

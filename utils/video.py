from __future__ import annotations

from datetime import datetime, timezone
import os
import shutil
import subprocess
import tempfile
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
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
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
    def __init__(self, camera_index: int, width: int, height: int, fps: int):
        self.cap = None
        self.picam2 = None
        self._prefetched_frame = None
        # Some Pi camera stacks already return BGR for 3-channel arrays.
        # Set PICAM_SWAP_RB=1 only if colors are still swapped.
        self._picam_swap_rb = os.getenv("PICAM_SWAP_RB", "0") == "1"

        # Prefer OpenCV for USB cameras and desktop environments.
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                self.cap = cap
                self._prefetched_frame = frame
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
            except Exception:
                if self.picam2 is not None:
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                self.picam2 = None

    def read(self) -> tuple[np.ndarray | None, datetime]:
        ts = datetime.now(timezone.utc)

        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            return frame, ts

        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return frame, ts

        if self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                if frame is None:
                    return None, ts
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    if self._picam_swap_rb:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    if self._picam_swap_rb:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                return frame, ts
            except Exception:
                return None, ts

        return None, ts

    def release(self) -> None:
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

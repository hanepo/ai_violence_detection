from __future__ import annotations

import time


class Buzzer:
    """Simple GPIO buzzer helper with mock fallback for non-RPi environments."""

    def __init__(self, pin: int = 18, enabled: bool = False, use_mock: bool = True):
        self.pin = pin
        self.enabled = enabled
        self.use_mock = use_mock
        self._gpio = None

        if not self.enabled:
            return

        if not use_mock:
            try:
                import RPi.GPIO as GPIO  # type: ignore

                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.OUT)
                GPIO.output(self.pin, GPIO.LOW)
                self._gpio = GPIO
            except Exception:
                self.use_mock = True

    def beep_pattern(self, repeats: int = 3, on_seconds: float = 0.15, off_seconds: float = 0.08) -> None:
        if not self.enabled or self._gpio is None:
            return

        for _ in range(max(1, int(repeats))):
            self._gpio.output(self.pin, self._gpio.HIGH)
            time.sleep(max(0.01, on_seconds))
            self._gpio.output(self.pin, self._gpio.LOW)
            time.sleep(max(0.01, off_seconds))

    def cleanup(self) -> None:
        if self._gpio is None:
            return
        try:
            self._gpio.output(self.pin, self._gpio.LOW)
        except Exception:
            pass
        try:
            self._gpio.cleanup(self.pin)
        except Exception:
            pass

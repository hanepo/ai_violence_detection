from __future__ import annotations

import random
from time import monotonic


class PIRSensor:
    """GPIO abstraction with optional mock mode for Mac development."""

    def __init__(self, pin: int = 17, use_mock: bool = True):
        self.pin = pin
        self.use_mock = use_mock
        self._last_trigger = monotonic()

        self._gpio = None
        if not use_mock:
            try:
                import RPi.GPIO as GPIO  # type: ignore

                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.IN)
                self._gpio = GPIO
            except Exception:
                self.use_mock = True

    def motion_detected(self) -> bool:
        if self.use_mock:
            # Simulate occasional motion to exercise the full pipeline on Mac.
            if monotonic() - self._last_trigger > 8 and random.random() < 0.25:
                self._last_trigger = monotonic()
                return True
            return False

        if self._gpio is None:
            return False
        return bool(self._gpio.input(self.pin))

    def cleanup(self) -> None:
        if self._gpio is not None:
            self._gpio.cleanup()

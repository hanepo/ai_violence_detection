from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Any

import paho.mqtt.client as mqtt


class MQTTNotifier:
    def __init__(self, broker: str, port: int, topic: str):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self._connected = False

    def connect(self) -> bool:
        if self._connected:
            return True
        try:
            self.client.connect(self.broker, self.port, keepalive=30)
            self.client.loop_start()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def publish_alert(self, payload: Dict[str, Any]) -> bool:
        if not self._connected and not self.connect():
            return False
        try:
            self.client.publish(self.topic, json.dumps(payload), qos=1, retain=False)
            return True
        except Exception:
            self._connected = False
            return False

    @staticmethod
    def build_payload(device_id: str, threat_score: float, event_id: str) -> Dict[str, Any]:
        return {
            "device_id": device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threat_score": round(threat_score, 4),
            "event_id": event_id,
        }

    def close(self) -> None:
        if self._connected:
            self.client.loop_stop()
            self.client.disconnect()
            self._connected = False

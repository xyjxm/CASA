"""ZMQ packed-message publisher used by CASA and SONIC smoke tasks."""

from __future__ import annotations

import json
import struct
import time
from typing import Any


HEADER_SIZE = 1280
LOCOMOTION_IDLE = 0
LOCOMOTION_WALK = 2


class ZMQPlannerPublisher:
    """Publish command and planner messages in the deploy ZMQManager format."""

    def __init__(self, host: str, port: int, dry_run: bool = False) -> None:
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self.socket = None
        self.context = None
        self.sent_messages: list[dict[str, Any]] = []
        if dry_run:
            return

        import zmq  # Imported lazily so --dry-run works in minimal envs.

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{host}:{port}")
        time.sleep(0.5)

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(0)
        if self.context is not None:
            self.context.term()

    def send_command(
        self,
        *,
        start: bool,
        stop: bool,
        planner: bool,
        delta_heading: float | None = None,
    ) -> None:
        fields = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        data = bytearray()
        data.extend(struct.pack("B", 1 if start else 0))
        data.extend(struct.pack("B", 1 if stop else 0))
        data.extend(struct.pack("B", 1 if planner else 0))
        if delta_heading is not None:
            fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
            data.extend(struct.pack("<f", delta_heading))

        self._send_packed("command", fields, bytes(data))
        self.sent_messages.append(
            {
                "topic": "command",
                "start": start,
                "stop": stop,
                "planner": planner,
                "delta_heading": delta_heading,
            }
        )

    def send_planner(
        self,
        *,
        mode: int,
        movement: tuple[float, float, float],
        facing: tuple[float, float, float],
        speed: float = -1.0,
        height: float = -1.0,
        upper_body_position: list[float] | None = None,
        upper_body_velocity: list[float] | None = None,
    ) -> None:
        fields = [
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
        ]
        data = bytearray()
        data.extend(struct.pack("<i", mode))
        data.extend(struct.pack("<fff", *movement))
        data.extend(struct.pack("<fff", *facing))
        data.extend(struct.pack("<f", speed))
        data.extend(struct.pack("<f", height))
        if upper_body_position is not None:
            fields.append({"name": "upper_body_position", "dtype": "f32", "shape": [17]})
            data.extend(struct.pack("<17f", *upper_body_position))
        if upper_body_velocity is not None:
            fields.append({"name": "upper_body_velocity", "dtype": "f32", "shape": [17]})
            data.extend(struct.pack("<17f", *upper_body_velocity))

        self._send_packed("planner", fields, bytes(data))
        self.sent_messages.append(
            {
                "topic": "planner",
                "mode": mode,
                "movement": movement,
                "facing": facing,
                "speed": speed,
                "height": height,
                "has_upper_body": upper_body_position is not None,
            }
        )

    def _pack_message(self, topic: str, fields: list[dict[str, Any]], data: bytes) -> bytes:
        header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        if len(header_json) > HEADER_SIZE:
            raise ValueError(f"ZMQ packed header exceeds {HEADER_SIZE} bytes")
        return topic.encode("utf-8") + header_json + (b"\x00" * (HEADER_SIZE - len(header_json))) + data

    def _send_packed(self, topic: str, fields: list[dict[str, Any]], data: bytes) -> None:
        message = self._pack_message(topic, fields, data)
        if self.socket is not None:
            self.socket.send(message)


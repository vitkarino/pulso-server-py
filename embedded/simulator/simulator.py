from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import signal
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed


@dataclass(frozen=True)
class MeasurementProfile:
    bpm: float
    spo2_ratio: float
    temperature: float
    noise: float
    ir_dc: float
    red_dc: float
    ir_ac: float
    phase_offset: float


@dataclass(frozen=True)
class SimulatorConfig:
    host: str
    port: int
    device_id: str
    fs: float
    batch_size: int
    bpm: float | None
    spo2_ratio: float | None
    temperature: float | None
    noise: float | None
    auto_start: bool
    auto_duration_s: float | None
    once: bool

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws/devices/{_public_device_id(self.device_id)}"


class PulsoDeviceSimulator:
    def __init__(self, config: SimulatorConfig) -> None:
        self._config = config
        self._stop_event = asyncio.Event()
        self._active_task: asyncio.Task[None] | None = None
        self._active_measurement_id: str | None = None
        self._active_profile: MeasurementProfile | None = None
        self._completed_once = False

    async def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self._config.url) as websocket:
                    print(f"connected: {self._config.url}", flush=True)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "hello",
                                "device_id": _public_device_id(self._config.device_id),
                                "is_simulated": True,
                            }
                        )
                    )
                    if self._config.auto_start:
                        await self._start_stream(
                            websocket,
                            measurement_id=f"sim-{self._config.device_id}",
                            duration_s=self._config.auto_duration_s,
                        )
                    await self._receive_loop(websocket)
                    if self._config.once and self._completed_once:
                        break
            except (OSError, ConnectionClosed) as exc:
                if self._stop_event.is_set():
                    break
                if self._config.once and self._completed_once:
                    break
                print(f"connection lost: {exc}; reconnecting in 2s", flush=True)
                await asyncio.sleep(2)

    def stop(self) -> None:
        self._stop_event.set()
        if self._active_task is not None:
            self._active_task.cancel()

    async def _receive_loop(self, websocket: websockets.ClientConnection) -> None:
        async for raw_message in websocket:
            message = _decode_json(raw_message)
            if message is None:
                await self._send_log(websocket, "received non-json message")
                continue

            message_type = message.get("type")
            if message_type == "hello_ack":
                print("server acknowledged device channel", flush=True)
                continue
            if message_type == "start":
                measurement_id = message.get("measurement_id")
                if not isinstance(measurement_id, str) or not measurement_id:
                    await self._send_log(websocket, "start command without measurement_id")
                    continue
                duration_s = _positive_float(message.get("duration"))
                print(
                    f"start command received: measurement_id={measurement_id}, duration_s={duration_s}",
                    flush=True,
                )
                await self._start_stream(websocket, measurement_id, duration_s)
                continue
            if message_type == "stop":
                print("stop command received", flush=True)
                await self._stop_stream(websocket)
                continue

            print(f"ignored server message type: {message_type}", flush=True)
            await self._send_log(websocket, f"ignored message type: {message_type}")

    async def _start_stream(
        self,
        websocket: websockets.ClientConnection,
        measurement_id: str,
        duration_s: float | None,
    ) -> None:
        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()
        self._active_measurement_id = measurement_id
        self._active_profile = self._new_profile()
        await websocket.send(
            json.dumps(
                {
                    "type": "start_ack",
                    "device_id": _public_device_id(self._config.device_id),
                    "measurement_id": measurement_id,
                }
            )
        )
        self._active_task = asyncio.create_task(
            self._stream_measurement(websocket, measurement_id, duration_s)
        )

    async def _stop_stream(self, websocket: websockets.ClientConnection) -> None:
        stopped_measurement_id = self._active_measurement_id
        if stopped_measurement_id is None:
            await websocket.send(
                json.dumps(
                    {
                        "type": "stop_ack",
                        "device_id": _public_device_id(self._config.device_id),
                    }
                )
            )
            print("stop command ignored: stream is already idle", flush=True)
            return

        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()
        self._active_task = None
        self._active_measurement_id = None
        self._active_profile = None
        self._completed_once = True
        await websocket.send(
            json.dumps(
                {
                    "type": "stop_ack",
                    "device_id": _public_device_id(self._config.device_id),
                    "measurement_id": stopped_measurement_id,
                }
            )
        )
        print(f"stream stopped by server: measurement_id={stopped_measurement_id}", flush=True)
        if self._config.once:
            self.stop()

    async def _stream_measurement(
        self,
        websocket: websockets.ClientConnection,
        measurement_id: str,
        duration_s: float | None,
    ) -> None:
        total_samples = (
            max(1, int(math.ceil(duration_s * self._config.fs - 1e-9)))
            if duration_s is not None
            else None
        )
        sent_samples = 0
        profile = self._active_profile or self._new_profile()
        print(
            f"stream started: measurement_id={measurement_id}, duration_s={duration_s}, samples={total_samples or 'unbounded'}",
            flush=True,
        )
        print(
            "profile: "
            f"bpm={profile.bpm:.1f}, spo2_ratio={profile.spo2_ratio:.3f}, "
            f"temperature={profile.temperature:.1f}, noise={profile.noise:.1f}",
            flush=True,
        )

        try:
            while (total_samples is None or sent_samples < total_samples) and not self._stop_event.is_set():
                batch_count = self._config.batch_size
                if total_samples is not None:
                    batch_count = min(batch_count, total_samples - sent_samples)
                samples = [
                    self._sample(sample_index, profile)
                    for sample_index in range(sent_samples, sent_samples + batch_count)
                ]
                await websocket.send(
                    json.dumps(
                        {
                            "type": "samples",
                            "device_id": _public_device_id(self._config.device_id),
                            "measurement_id": measurement_id,
                            "recording_id": None,
                            "sample_rate_hz": self._config.fs,
                            "sensor_temp_c": profile.temperature,
                            "samples": samples,
                        }
                    )
                )
                sent_samples += batch_count
                total_label = total_samples if total_samples is not None else "unbounded"
                print(
                    f"sent batch: measurement_id={measurement_id}, samples={sent_samples}/{total_label}",
                    flush=True,
                )
                if total_samples is None or sent_samples < total_samples:
                    await asyncio.sleep(batch_count / self._config.fs)

            if total_samples is not None and not self._stop_event.is_set():
                await websocket.send(
                    json.dumps(
                        {
                            "type": "finished",
                            "device_id": _public_device_id(self._config.device_id),
                            "measurement_id": measurement_id,
                            "reason": "duration_reached",
                        }
                    )
                )
                print(f"stream finished: measurement_id={measurement_id}", flush=True)
                self._completed_once = True
                if self._config.once:
                    self.stop()
                    await websocket.close(code=1000, reason="measurement finished")
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            print("stream interrupted: websocket closed", flush=True)
        finally:
            if self._active_measurement_id == measurement_id:
                self._active_measurement_id = None
                self._active_task = None
                self._active_profile = None

    def _new_profile(self) -> MeasurementProfile:
        return MeasurementProfile(
            bpm=self._config.bpm if self._config.bpm is not None else random.uniform(58.0, 96.0),
            spo2_ratio=(
                self._config.spo2_ratio
                if self._config.spo2_ratio is not None
                else random.uniform(0.45, 0.68)
            ),
            temperature=(
                self._config.temperature
                if self._config.temperature is not None
                else random.uniform(30.5, 33.5)
            ),
            noise=self._config.noise if self._config.noise is not None else random.uniform(25.0, 90.0),
            ir_dc=random.uniform(78_000.0, 96_000.0),
            red_dc=random.uniform(48_000.0, 60_000.0),
            ir_ac=random.uniform(1_000.0, 2_200.0),
            phase_offset=random.uniform(0.0, 2.0 * math.pi),
        )

    def _sample(self, sample_index: int, profile: MeasurementProfile) -> dict[str, float]:
        t = sample_index / self._config.fs
        frequency_hz = profile.bpm / 60.0
        red_ac = profile.spo2_ratio * profile.ir_ac * profile.red_dc / profile.ir_dc
        phase = 2.0 * math.pi * frequency_hz * t + profile.phase_offset
        return {
            "index": sample_index,
            "ir": profile.ir_dc + profile.ir_ac * math.sin(phase) + random.gauss(0.0, profile.noise),
            "red": profile.red_dc + red_ac * math.sin(phase) + random.gauss(0.0, profile.noise * 0.75),
        }

    async def _send_log(self, websocket: websockets.ClientConnection, message: str) -> None:
        await websocket.send(
            json.dumps(
                {
                    "type": "log",
                    "device_id": _public_device_id(self._config.device_id),
                    "message": message,
                }
            )
        )


def parse_args() -> SimulatorConfig:
    parser = argparse.ArgumentParser(description="Pulso embedded device simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device-id", default="sim-device-001")
    parser.add_argument("--fs", type=float, default=400.0 / 3.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--spo2-ratio", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start streaming immediately without waiting for API start command.",
    )
    parser.add_argument("--auto-duration-s", type=float, default=None)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit after the first measurement is finished or stopped.",
    )
    args = parser.parse_args()
    return SimulatorConfig(
        host=args.host,
        port=args.port,
        device_id=args.device_id,
        fs=args.fs,
        batch_size=args.batch_size,
        bpm=args.bpm,
        spo2_ratio=args.spo2_ratio,
        temperature=args.temperature,
        noise=args.noise,
        auto_start=args.auto_start,
        auto_duration_s=args.auto_duration_s,
        once=args.once,
    )


def _decode_json(raw_message: str | bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_message)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _public_device_id(device_id: str) -> str:
    return device_id if device_id.startswith("dev_") else f"dev_{device_id}"


async def amain() -> None:
    simulator = PulsoDeviceSimulator(parse_args())
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, simulator.stop)
    await simulator.run()


if __name__ == "__main__":
    asyncio.run(amain())

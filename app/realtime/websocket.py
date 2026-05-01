import asyncio
import json
from threading import RLock
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.ids import internal_device_id, public_device_id
from app.measurements import RecordingEvent
from app.schemas.measurements import MeasurementState
from app.schemas.websocket import WebSocketAck
from app.processing.service import PPGProcessingService


class WebSocketController:
    def __init__(
        self,
        service: PPGProcessingService,
        *,
        metrics_formatter: Callable[[Any], object] | None = None,
        ack_timeout_seconds: float = 3.0,
    ) -> None:
        self._service = service
        self._ack_timeout_seconds = ack_timeout_seconds
        self._metrics_formatter = metrics_formatter or self._default_metrics_formatter
        self._lock = RLock()
        self._connections: dict[str, WebSocket] = {}
        self._measurement_subscribers: dict[str, set[WebSocket]] = {}
        self._pending_start_acks: dict[str, asyncio.Future[None]] = {}

    def connected_devices(self) -> list[str]:
        with self._lock:
            return sorted(self._connections)

    async def disconnect_device(self, device_id: str) -> bool:
        with self._lock:
            websocket = self._connections.pop(device_id, None)
        if websocket is None:
            return False

        try:
            await websocket.close()
        except RuntimeError:
            pass
        return True

    async def send_start(
        self,
        *,
        device_id: str,
        measurement_id: str,
        duration_seconds: float | None,
    ) -> bool:
        websocket = self._connection_for(device_id)
        if websocket is None:
            return False

        loop = asyncio.get_running_loop()
        start_ack = loop.create_future()
        payload: dict[str, Any] = {
            "type": "start",
            "measurement_id": measurement_id,
        }
        if duration_seconds is not None:
            payload["duration"] = duration_seconds

        try:
            with self._lock:
                self._pending_start_acks[measurement_id] = start_ack
            await websocket.send_json(payload)
            await asyncio.wait_for(start_ack, timeout=self._ack_timeout_seconds)
        except RuntimeError:
            self._unregister(device_id, websocket)
            return False
        except asyncio.TimeoutError:
            self._unregister(device_id, websocket)
            return False
        finally:
            with self._lock:
                if self._pending_start_acks.get(measurement_id) is start_ack:
                    del self._pending_start_acks[measurement_id]
        return True

    async def send_stop(self, *, device_id: str) -> bool:
        websocket = self._connection_for(device_id)
        if websocket is None:
            return False

        try:
            await websocket.send_json({"type": "stop"})
        except RuntimeError:
            self._unregister(device_id, websocket)
            return False
        return True

    async def handle_device(self, websocket: WebSocket, device_id: str) -> None:
        await websocket.accept()
        with self._lock:
            self._connections[device_id] = websocket
        await websocket.send_json(
            {
                "type": "hello_ack",
                "device_id": public_device_id(device_id),
                "connection_status": "connected",
            }
        )
        await self._run_device_loop(websocket, device_id=device_id)

    async def _run_device_loop(self, websocket: WebSocket, device_id: str | None) -> None:
        try:
            while True:
                message = await websocket.receive_text()
                hello_device_id = self._handle_hello(websocket, message)
                if hello_device_id is not None:
                    device_id = hello_device_id
                    await websocket.send_json(
                        {
                            "type": "hello_ack",
                            "device_id": public_device_id(device_id),
                            "connection_status": "connected",
                        }
                    )
                    continue
                if await self._handle_device_control_message(message, device_id):
                    continue

                try:
                    processed = self._service.process_json_with_recordings_and_signal(message)
                    metrics = processed.metrics
                    completed_measurements = processed.completed_measurements
                    recording_events = processed.recording_events
                    ack = WebSocketAck(ok=True, metrics=metrics)
                    await self._broadcast_live_sample_batch(
                        message,
                        metrics,
                        processed_samples=processed.processed_samples,
                    )
                except ValidationError as exc:
                    ack = WebSocketAck(ok=False, error=exc.errors()[0]["msg"])
                    completed_measurements = []
                    recording_events = []
                except Exception as exc:
                    ack = WebSocketAck(ok=False, error=str(exc))
                    completed_measurements = []
                    recording_events = []

                if device_id is None or not ack.ok:
                    await websocket.send_json(ack.model_dump(mode="json"))
                for event in recording_events:
                    await self.broadcast_recording_event(event)
                for measurement in completed_measurements:
                    if measurement.device_id is not None:
                        await self.send_stop(device_id=measurement.device_id)
                    state = self._service.get_recording_state(measurement.measurement_id)
                    if state is not None:
                        await self.broadcast_measurement_finished(state)
        except WebSocketDisconnect:
            if device_id is not None:
                self._service.stop_recording_for_device(device_id)
                self._unregister(device_id, websocket)
            return

    async def handle_measurement_stream(self, websocket: WebSocket, measurement_id: str) -> None:
        await websocket.accept()
        with self._lock:
            subscribers = self._measurement_subscribers.setdefault(measurement_id, set())
            subscribers.add(websocket)
        try:
            await websocket.send_json(
                {
                    "type": "live_ack",
                    "ok": True,
                    "measurement_id": measurement_id,
                }
            )
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            self._unregister_measurement_stream(websocket)
        except RuntimeError:
            self._unregister_measurement_stream(websocket)

    def _handle_hello(self, websocket: WebSocket, message: str) -> str | None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict) or payload.get("type") != "hello":
            return None

        device_id = internal_device_id(payload.get("device_id"))
        if not isinstance(device_id, str) or not device_id:
            return None

        with self._lock:
            self._connections[device_id] = websocket
        return device_id

    async def _handle_device_control_message(self, message: str, device_id: str | None) -> bool:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return False

        if not isinstance(payload, dict):
            return False
        message_type = payload.get("type")
        if message_type == "start_ack":
            measurement_id = payload.get("measurement_id")
            if isinstance(measurement_id, str):
                self._ack_start(measurement_id)
            return True
        if message_type in {"stop_ack", "log"}:
            return True
        if message_type != "finished":
            return False

        measurement_id = payload.get("measurement_id")
        if isinstance(measurement_id, str) and measurement_id:
            stopped = self._service.stop_measurement(measurement_id)
            if stopped is not None:
                measurement, event = stopped
                if event is not None:
                    await self.broadcast_recording_event(event)
                await self.broadcast_measurement_finished(measurement)
        elif device_id is not None:
            stopped_state = self._service.stop_recording_for_device(device_id)
            if stopped_state is not None:
                await self.broadcast_measurement_finished(stopped_state)
        return True

    def _connection_for(self, device_id: str) -> WebSocket | None:
        with self._lock:
            return self._connections.get(device_id)

    def _unregister(self, device_id: str, websocket: WebSocket) -> None:
        with self._lock:
            if self._connections.get(device_id) is websocket:
                del self._connections[device_id]

    def _ack_start(self, measurement_id: str) -> None:
        with self._lock:
            start_ack = self._pending_start_acks.get(measurement_id)
        if start_ack is not None and not start_ack.done():
            start_ack.set_result(None)

    async def _broadcast_live_sample_batch(
        self,
        message: str | bytes,
        metrics: object,
        *,
        processed_samples: list[dict[str, float]] | None = None,
    ) -> None:
        payload = self._live_sample_payload(message, metrics, processed_samples=processed_samples)
        if payload is None:
            return

        with self._lock:
            measurement_id = payload.get("measurement_id")
            measurement_subscribers = (
                self._measurement_subscribers.get(measurement_id, set())
                if isinstance(measurement_id, str)
                else set()
            )
            subscribers = list(measurement_subscribers)

        stale: list[WebSocket] = []
        for subscriber in subscribers:
            try:
                await subscriber.send_json(payload)
            except RuntimeError:
                stale.append(subscriber)

        for subscriber in stale:
            self._unregister_measurement_stream(subscriber)

    async def broadcast_recording_event(self, event: RecordingEvent) -> None:
        payload: dict[str, Any]
        if event.type == "recording_started":
            payload = {
                "type": "recording_started",
                "measurement_id": event.measurement_id,
                "recording_id": event.recording_id,
                "sample_range": {
                    "start_index": event.sample_start_index,
                    "end_index": None,
                },
                "timestamp": self._timestamp(),
            }
        else:
            payload = {
                "type": "recording_stopped",
                "measurement_id": event.measurement_id,
                "recording_id": event.recording_id,
                "sample_range": {
                    "start_index": event.sample_start_index,
                    "end_index": event.sample_end_index,
                },
                "samples_count": event.samples_count,
                "timestamp": self._timestamp(),
            }
        await self._broadcast_to_measurement(event.measurement_id, payload)

    async def broadcast_measurement_finished(self, measurement: MeasurementState) -> None:
        if measurement.id is None:
            return
        await self._broadcast_to_measurement(
            measurement.id,
            {
                "type": "measurement_finished",
                "measurement_id": measurement.id,
                "status": measurement.status,
                "timestamp": self._timestamp(),
            },
        )

    async def _broadcast_to_measurement(self, measurement_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._measurement_subscribers.get(measurement_id, set()))

        stale: list[WebSocket] = []
        for subscriber in subscribers:
            try:
                await subscriber.send_json(payload)
            except RuntimeError:
                stale.append(subscriber)
        for subscriber in stale:
            self._unregister_measurement_stream(subscriber)

    def _live_sample_payload(
        self,
        message: str | bytes,
        metrics: object,
        *,
        processed_samples: list[dict[str, float]] | None = None,
    ) -> dict[str, Any] | None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None
        device = payload.get("device")
        if not isinstance(device, dict):
            if payload.get("type") == "samples":
                device = {
                    "id": payload.get("device_id"),
                    "measurement_id": payload.get("measurement_id"),
                    "recording_id": payload.get("recording_id"),
                    "fs": payload.get("sample_rate_hz"),
                    "temp": payload.get("sensor_temp_c"),
                    "samples": payload.get("samples"),
                }
            else:
                return None

        measurement_id = device.get("measurement_id")
        if not isinstance(measurement_id, str) or not measurement_id:
            return None

        samples = device.get("samples")
        if not isinstance(samples, list):
            return None
        sample_rate = device.get("fs")
        output_samples = self._samples_for_client(
            raw_samples=samples,
            processed_samples=processed_samples,
            sample_rate=sample_rate if isinstance(sample_rate, (int, float)) else None,
        )
        state = self._service.get_recording_state(measurement_id)

        return {
            "type": "measurement_update",
            "measurement_id": measurement_id,
            "device_id": public_device_id(device.get("id")),
            "active_recording_id": state.active_recording_id if state is not None else device.get("recording_id"),
            "timestamp": self._timestamp(),
            "sample_rate_hz": device.get("fs"),
            "sensor_temp_c": device.get("temp"),
            "samples": output_samples,
            "metrics": self._metrics_formatter(metrics),
        }

    @staticmethod
    def _samples_for_client(
        *,
        raw_samples: list[Any],
        processed_samples: list[dict[str, float]] | None,
        sample_rate: float | None,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        processed_samples = processed_samples or []
        for index, raw_sample in enumerate(raw_samples):
            raw = raw_sample if isinstance(raw_sample, dict) else {}
            processed = processed_samples[index] if index < len(processed_samples) else {}
            sample_index = raw.get("index", index)
            t_ms = None
            if isinstance(sample_index, (int, float)) and sample_rate:
                t_ms = round(float(sample_index) / sample_rate * 1000.0, 3)
            output.append(
                {
                    "index": sample_index,
                    "t_ms": t_ms,
                    "ir": raw.get("ir"),
                    "red": raw.get("red", raw.get("r")),
                    "ir_filtered": processed.get("ir"),
                    "red_filtered": processed.get("red", processed.get("r")),
                }
            )
        return output

    @staticmethod
    def _timestamp() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _default_metrics_formatter(metrics: Any) -> object:
        if hasattr(metrics, "model_dump"):
            return metrics.model_dump(mode="json")
        return metrics

    def _unregister_measurement_stream(self, websocket: WebSocket) -> None:
        with self._lock:
            empty_measurements: list[str] = []
            for measurement_id, subscribers in self._measurement_subscribers.items():
                subscribers.discard(websocket)
                if not subscribers:
                    empty_measurements.append(measurement_id)
            for measurement_id in empty_measurements:
                del self._measurement_subscribers[measurement_id]

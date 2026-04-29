import asyncio
import json
from threading import RLock
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

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
        await websocket.send_json({"ok": True, "type": "hello_ack", "device_id": device_id})
        await self._run_device_loop(websocket, device_id=device_id)

    async def _run_device_loop(self, websocket: WebSocket, device_id: str | None) -> None:
        try:
            while True:
                message = await websocket.receive_text()
                hello_device_id = self._handle_hello(websocket, message)
                if hello_device_id is not None:
                    device_id = hello_device_id
                    await websocket.send_json({"ok": True, "type": "hello_ack"})
                    continue
                if self._handle_device_control_message(message, device_id):
                    continue

                try:
                    processed = self._service.process_json_with_recordings_and_signal(message)
                    metrics = processed.metrics
                    completed_recordings = processed.completed_recordings
                    ack = WebSocketAck(ok=True, metrics=metrics)
                    await self._broadcast_live_sample_batch(
                        message,
                        metrics,
                        processed_samples=processed.processed_samples,
                    )
                except ValidationError as exc:
                    ack = WebSocketAck(ok=False, error=exc.errors()[0]["msg"])
                    completed_recordings = []
                except Exception as exc:
                    ack = WebSocketAck(ok=False, error=str(exc))
                    completed_recordings = []

                if device_id is None or not ack.ok:
                    await websocket.send_json(ack.model_dump(mode="json"))
                for recording in completed_recordings:
                    if recording.device_id is not None:
                        await self.send_stop(device_id=recording.device_id)
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

        device_id = payload.get("device_id")
        if not isinstance(device_id, str) or not device_id:
            return None

        with self._lock:
            self._connections[device_id] = websocket
        return device_id

    def _handle_device_control_message(self, message: str, device_id: str | None) -> bool:
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
            self._service.stop_recording(measurement_id)
        elif device_id is not None:
            self._service.stop_recording_for_device(device_id)
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
            return None

        measurement_id = device.get("measurement_id")
        if not isinstance(measurement_id, str) or not measurement_id:
            return None

        samples = device.get("samples")
        if not isinstance(samples, list):
            return None
        output_samples = processed_samples if processed_samples is not None else samples

        return {
            "type": "measurement_sample_batch",
            "measurement_id": measurement_id,
            "device_id": device.get("id"),
            "fs": device.get("fs"),
            "temperature": device.get("temp"),
            "sample_count": len(output_samples),
            "samples": output_samples,
            "metrics": self._metrics_formatter(metrics),
        }

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

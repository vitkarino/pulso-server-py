from __future__ import annotations

import json
from threading import RLock
from typing import Any
from uuid import uuid4

from app.core.config import AppConfig
from app.measurements.models import CompletedRecording, RecordingMetadata
from app.measurements.session import MeasurementSession
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator
from app.schemas.device import DeviceData
from app.schemas.measurements import MeasurementState
from app.schemas.metrics import VitalSigns
from app.storage.recording_repository import RecordingRepository


class MeasurementManager:
    def __init__(
        self,
        config: AppConfig,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
        repository: RecordingRepository | None = None,
    ) -> None:
        self._config = config
        self._filter = noise_filter
        self._calculator = calculator
        self._repository = repository
        self._lock = RLock()
        self._sessions: dict[str, MeasurementSession] = {}
        self._recording_by_device: dict[str, str] = {}

    def start(self, device_id: str, duration_seconds: float | None = None) -> MeasurementState:
        duration = duration_seconds or self._config.measurement_duration_seconds
        if duration < self._config.min_window_seconds:
            raise ValueError(
                f"measurement duration must be at least {self._config.min_window_seconds} seconds"
            )

        return self.start_recording(
            duration_seconds=duration,
            metadata=RecordingMetadata(),
            device_id=device_id,
        )

    def start_recording(
        self,
        *,
        duration_seconds: float | None,
        metadata: RecordingMetadata,
        device_id: str | None = None,
    ) -> MeasurementState:
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("recording duration must be greater than zero seconds")

        with self._lock:
            if device_id is not None:
                for existing in self._sessions.values():
                    if existing.device_id == device_id and existing.status == "running":
                        raise ValueError("measurement is already running for device")
            session = MeasurementSession(
                recording_id=str(uuid4()),
                device_id=device_id,
                duration_seconds=duration_seconds,
                metadata=metadata,
            )
            self._sessions[session.id] = session
            if device_id is not None:
                self._recording_by_device[device_id] = session.id
            snapshot = session.snapshot()

        self._create_recording(session, metadata)
        return snapshot

    def ingest(self, device: DeviceData) -> VitalSigns | None:
        completed = self.ingest_recordings(device)
        results = [recording.result for recording in completed if recording.result is not None]
        return results[-1] if results else None

    def ingest_recordings(self, device: DeviceData) -> list[CompletedRecording]:
        operations: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        completed: list[CompletedRecording] = []

        with self._lock:
            sessions = [
                session
                for session in self._sessions.values()
                if session.status == "running"
                and (session.device_id is None or session.device_id == device.id)
            ]
            for session in sessions:
                result, sample_rows, recording_update = session.append(
                    device=device,
                    noise_filter=self._filter,
                    calculator=self._calculator,
                )
                if session.device_id is not None:
                    self._recording_by_device[session.device_id] = session.id
                if sample_rows or recording_update:
                    operations.append((session.id, sample_rows, recording_update))
                if result is not None:
                    completed.append(
                        CompletedRecording(
                            recording_id=session.id,
                            device_id=session.device_id,
                            result=result,
                        )
                    )

        self._persist_operations(operations)
        for recording in completed:
            if recording.result is not None:
                self._print_result(recording.result)
        return completed

    def stop_recording(self, recording_id: str) -> MeasurementState | None:
        with self._lock:
            session = self._sessions.get(recording_id)
            if session is None or session.status != "running":
                return None
            result = session.stop(self._filter, self._calculator)
            snapshot = session.snapshot()
            operation = (session.id, [], session.recording_update())

        self._persist_operations([operation])
        if result is not None:
            self._print_result(result)
        return snapshot

    def get_recording_state(self, recording_id: str) -> MeasurementState | None:
        with self._lock:
            session = self._sessions.get(recording_id)
            return session.snapshot() if session is not None else None

    def stop_recording_for_device(self, device_id: str) -> MeasurementState | None:
        with self._lock:
            recording_id = self._recording_by_device.get(device_id)
            if recording_id is None:
                return None

        return self.stop_recording(recording_id)

    def delete_recording(self, recording_id: str) -> MeasurementState | None:
        with self._lock:
            session = self._sessions.pop(recording_id, None)
            if session is not None and session.device_id is not None:
                if self._recording_by_device.get(session.device_id) == recording_id:
                    del self._recording_by_device[session.device_id]
            snapshot = session.snapshot() if session is not None else None

        if self._repository is not None and self._repository.enabled:
            deleted = self._repository.delete_recording(recording_id)
            if not deleted and snapshot is None:
                return None
        return snapshot

    def stop_all(self) -> list[MeasurementState]:
        operations: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        results: list[VitalSigns] = []
        snapshots: list[MeasurementState] = []

        with self._lock:
            for session in self._sessions.values():
                if session.status != "running":
                    continue
                result = session.stop(self._filter, self._calculator)
                snapshots.append(session.snapshot())
                operations.append((session.id, [], session.recording_update()))
                if result is not None:
                    results.append(result)

        self._persist_operations(operations)
        for result in results:
            self._print_result(result)
        return snapshots

    def get(self, device_id: str) -> MeasurementState | None:
        with self._lock:
            recording_id = self._recording_by_device.get(device_id)
            if recording_id is not None:
                session = self._sessions.get(recording_id)
                if session is not None:
                    return session.snapshot()

            for session in self._sessions.values():
                if session.device_id == device_id:
                    return session.snapshot()
            return None

    def all(self) -> dict[str, MeasurementState]:
        with self._lock:
            return {
                session.device_id or session.id: session.snapshot()
                for session in self._sessions.values()
            }

    def _create_recording(self, session: MeasurementSession, metadata: RecordingMetadata) -> None:
        if self._repository is None or not self._repository.enabled:
            return
        self._repository.create_recording(session.initial_recording_values(metadata))

    def _persist_operations(
        self,
        operations: list[tuple[str, list[dict[str, Any]], dict[str, Any]]],
    ) -> None:
        if self._repository is None or not self._repository.enabled:
            return
        for recording_id, sample_rows, recording_update in operations:
            self._repository.insert_samples(recording_id, sample_rows)
            self._repository.update_recording(recording_id, recording_update)

    @staticmethod
    def _print_result(result: VitalSigns) -> None:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False), flush=True)

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import RLock
from typing import Any

from app.core.ids import (
    MEASUREMENT_PREFIX,
    RECORDING_PREFIX,
    new_internal_id,
    new_public_id,
    public_project_id,
    public_user_id,
)
from app.core.config import AppConfig
from app.measurements.models import CompletedMeasurement, RecordingEvent, RecordingMetadata
from app.measurements.session import MeasurementSession
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator
from app.schemas.device import DeviceData
from app.schemas.measurements import MeasurementState
from app.schemas.metrics import VitalSigns
from app.storage.recording_repository import RecordingRepository


@dataclass(frozen=True)
class IngestResult:
    completed_measurements: list[CompletedMeasurement]
    recording_events: list[RecordingEvent]


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
        self._measurement_by_device: dict[str, str] = {}

    def start(
        self,
        device_id: str,
        duration_seconds: float | None = None,
        metadata: RecordingMetadata | None = None,
    ) -> MeasurementState:
        if duration_seconds is not None and duration_seconds < self._config.min_window_seconds:
            raise ValueError(
                f"measurement duration must be at least {self._config.min_window_seconds} seconds"
            )
        return self.start_measurement(
            duration_seconds=duration_seconds,
            metadata=metadata or RecordingMetadata(),
            device_id=device_id,
        )

    def start_measurement(
        self,
        *,
        duration_seconds: float | None,
        metadata: RecordingMetadata,
        device_id: str | None = None,
    ) -> MeasurementState:
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("measurement duration must be greater than zero seconds")

        normalized_metadata = RecordingMetadata(
            user_name=metadata.user_name,
            user_id=public_user_id(metadata.user_id),
            project_name=metadata.project_name,
            project_id=public_project_id(metadata.project_id),
        )

        with self._lock:
            if device_id is not None:
                for existing in self._sessions.values():
                    if existing.device_id == device_id and existing.status == "running":
                        raise ValueError("measurement is already running for device")
            measurement_id = new_internal_id()
            session = MeasurementSession(
                measurement_id=measurement_id,
                public_measurement_id=new_public_id(MEASUREMENT_PREFIX, measurement_id),
                device_id=device_id,
                duration_seconds=duration_seconds,
                metadata=normalized_metadata,
            )
            self._sessions[session.internal_id] = session
            if device_id is not None:
                self._measurement_by_device[device_id] = session.internal_id
            snapshot = session.snapshot()

        self._create_measurement(session)
        return snapshot

    def start_recording(
        self,
        *,
        duration_seconds: float | None,
        metadata: RecordingMetadata,
        device_id: str | None = None,
    ) -> MeasurementState:
        snapshot = self.start_measurement(
            duration_seconds=duration_seconds,
            metadata=metadata,
            device_id=device_id,
        )
        if snapshot.id is not None:
            self.start_recording_for_measurement(snapshot.id, duration_seconds=duration_seconds)
        return self.get_recording_state(snapshot.id or "") or snapshot

    def start_recording_for_measurement(
        self,
        measurement_id: str,
        *,
        duration_seconds: float | None,
    ) -> tuple[MeasurementState, RecordingEvent]:
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("recording duration must be greater than zero seconds")

        with self._lock:
            session = self._session_by_id(measurement_id)
            if session is None:
                raise KeyError("measurement not found")
            recording_id = new_internal_id()
            recording = session.start_recording(
                recording_id=recording_id,
                public_recording_id=new_public_id(RECORDING_PREFIX, recording_id),
                duration_seconds=duration_seconds,
            )
            snapshot = session.snapshot()
            measurement_update = session.measurement_update()
            recording_values = session.initial_recording_values(recording)
            event = RecordingEvent(
                type="recording_started",
                measurement_id=session.public_id,
                recording_id=recording.public_id,
                sample_start_index=recording.sample_start_index,
                sample_end_index=None,
                samples_count=None,
            )

        self._create_recording(recording_values)
        self._update_measurement(session.internal_id, measurement_update)
        return snapshot, event

    def ingest(self, device: DeviceData) -> VitalSigns | None:
        result = self.ingest_recordings(device)
        completed = [
            measurement.result
            for measurement in result.completed_measurements
            if measurement.result is not None
        ]
        return completed[-1] if completed else None

    def ingest_recordings(self, device: DeviceData) -> IngestResult:
        operations: list[tuple[str, dict[str, Any], str | None, list[dict[str, Any]], dict[str, Any] | None]] = []
        completed: list[CompletedMeasurement] = []
        recording_events: list[RecordingEvent] = []

        with self._lock:
            sessions = [
                session
                for session in self._sessions.values()
                if session.status == "running"
                and (session.device_id is None or session.device_id == device.id)
            ]
            for session in sessions:
                result = session.append(
                    device=device,
                    noise_filter=self._filter,
                    calculator=self._calculator,
                )
                if not result.measurement_update:
                    continue
                if session.device_id is not None:
                    self._measurement_by_device[session.device_id] = session.internal_id
                operations.append(
                    (
                        session.internal_id,
                        result.measurement_update,
                        result.recording_id,
                        result.sample_rows,
                        result.recording_update,
                    )
                )
                recording_events.extend(result.events)
                if result.measurement_finished:
                    completed.append(
                        CompletedMeasurement(
                            measurement_id=session.public_id,
                            device_id=session.device_id,
                            result=result.result,
                        )
                    )

        self._persist_operations(operations)
        for measurement in completed:
            if measurement.result is not None:
                self._print_result(measurement.result)
        return IngestResult(completed_measurements=completed, recording_events=recording_events)

    def stop_measurement(self, measurement_id: str) -> tuple[MeasurementState, RecordingEvent | None] | None:
        with self._lock:
            session = self._session_by_id(measurement_id)
            if session is None or session.status != "running":
                return None
            result, recording_update, event = session.stop_measurement(self._filter, self._calculator)
            snapshot = session.snapshot()
            operation = (
                session.internal_id,
                session.measurement_update(),
                event.recording_id if event is not None else None,
                [],
                recording_update,
            )

        self._persist_operations([operation])
        if result is not None:
            self._print_result(result)
        return snapshot, event

    def stop_recording_for_measurement(self, measurement_id: str) -> tuple[MeasurementState, RecordingEvent] | None:
        with self._lock:
            session = self._session_by_id(measurement_id)
            if session is None or session.status != "running":
                return None
            recording_update, event = session.stop_recording(self._filter, self._calculator)
            if event is None:
                return None
            snapshot = session.snapshot()
            operation = (
                session.internal_id,
                session.measurement_update(),
                event.recording_id,
                [],
                recording_update,
            )

        self._persist_operations([operation])
        return snapshot, event

    def get_recording_state(self, measurement_id: str) -> MeasurementState | None:
        with self._lock:
            session = self._session_by_id(measurement_id)
            return session.snapshot() if session is not None else None

    def stop_recording(self, measurement_id: str) -> MeasurementState | None:
        stopped = self.stop_measurement(measurement_id)
        return stopped[0] if stopped is not None else None

    def stop_recording_for_device(self, device_id: str) -> MeasurementState | None:
        with self._lock:
            measurement_id = self._measurement_by_device.get(device_id)
            if measurement_id is None:
                return None
        return self.stop_recording(measurement_id)

    def delete_recording(self, recording_id: str) -> MeasurementState | None:
        with self._lock:
            snapshot = None
            for session in self._sessions.values():
                active = session.active_recording
                if active is not None and recording_id in {active.id, active.public_id}:
                    self.stop_recording_for_measurement(session.public_id)
                    snapshot = session.snapshot()
                    break

        if self._repository is not None and self._repository.enabled:
            deleted = self._repository.delete_recording(recording_id)
            if not deleted and snapshot is None:
                return None
        return snapshot

    def stop_all(self) -> list[MeasurementState]:
        snapshots: list[MeasurementState] = []
        with self._lock:
            measurement_ids = [
                session.public_id
                for session in self._sessions.values()
                if session.status == "running"
            ]
        for measurement_id in measurement_ids:
            stopped = self.stop_measurement(measurement_id)
            if stopped is not None:
                snapshots.append(stopped[0])
        return snapshots

    def get(self, device_id: str) -> MeasurementState | None:
        with self._lock:
            measurement_id = self._measurement_by_device.get(device_id)
            if measurement_id is not None:
                session = self._sessions.get(measurement_id)
                if session is not None:
                    return session.snapshot()

            for session in self._sessions.values():
                if session.device_id == device_id:
                    return session.snapshot()
            return None

    def all(self) -> dict[str, MeasurementState]:
        with self._lock:
            return {
                session.device_id or session.public_id: session.snapshot()
                for session in self._sessions.values()
            }

    def _create_measurement(self, session: MeasurementSession) -> None:
        if self._repository is None or not self._repository.enabled:
            return
        self._repository.create_measurement(session.initial_measurement_values())

    def _create_recording(self, values: dict[str, Any]) -> None:
        if self._repository is None or not self._repository.enabled:
            return
        self._repository.create_recording(values)

    def _update_measurement(self, measurement_id: str, values: dict[str, Any]) -> None:
        if self._repository is None or not self._repository.enabled:
            return
        self._repository.update_measurement(measurement_id, values)

    def _persist_operations(
        self,
        operations: list[tuple[str, dict[str, Any], str | None, list[dict[str, Any]], dict[str, Any] | None]],
    ) -> None:
        if self._repository is None or not self._repository.enabled:
            return
        for measurement_id, measurement_update, recording_id, sample_rows, recording_update in operations:
            self._repository.update_measurement(measurement_id, measurement_update)
            if recording_id is not None:
                self._repository.insert_samples(recording_id, sample_rows)
                if recording_update is not None:
                    self._repository.update_recording(recording_id, recording_update)

    def _session_by_id(self, measurement_id: str) -> MeasurementSession | None:
        session = self._sessions.get(measurement_id)
        if session is not None:
            return session
        for candidate in self._sessions.values():
            if measurement_id in {candidate.public_id, candidate.internal_id}:
                return candidate
        return None

    @staticmethod
    def _print_result(result: VitalSigns) -> None:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False), flush=True)

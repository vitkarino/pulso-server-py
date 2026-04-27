from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from math import ceil
from threading import RLock
from typing import Any
from uuid import uuid4

import numpy as np

from app.config import AppConfig
from app.models import DeviceData, MeasurementState, PPGSample, VitalSigns
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator
from app.recording_repository import RecordingRepository


@dataclass(frozen=True)
class RecordingMetadata:
    user_name: str | None = None
    user_id: str | None = None
    project_name: str | None = None
    project_id: str | None = None


@dataclass(frozen=True)
class CompletedRecording:
    recording_id: str
    device_id: str | None
    result: VitalSigns | None


class MeasurementSession:
    def __init__(
        self,
        *,
        recording_id: str,
        device_id: str | None,
        duration_seconds: float | None,
    ) -> None:
        self.id = recording_id
        self.device_id = device_id
        self.duration_seconds = duration_seconds
        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None
        self.status = "running"
        self.reason: str | None = None
        self.result: VitalSigns | None = None
        self._fs: float | None = None
        self._temperature: float | None = None
        self._ir: list[float] = []
        self._red: list[float] = []
        self._next_sample_index = 0

    def append(
        self,
        device: DeviceData,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> tuple[VitalSigns | None, list[dict[str, Any]], dict[str, Any]]:
        if self.status != "running":
            return self.result, [], self.recording_update()

        if device.recording_id is not None and device.recording_id != self.id:
            return None, [], {}

        if self.device_id is None:
            self.device_id = device.id
        elif self.device_id != device.id:
            return None, [], {}

        if self._fs is None:
            self._fs = device.fs
        elif self._fs != device.fs:
            self._fail("sampling frequency changed during measurement")
            return None, [], self.recording_update()

        self._temperature = device.temp
        sample_rows = self._append_samples(self._samples_for_buffer_duration(device.samples))
        result: VitalSigns | None = None

        if self._duration_ready():
            result = self._finish(status="completed", noise_filter=noise_filter, calculator=calculator)

        return result, sample_rows, self.recording_update()

    def stop(
        self,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> VitalSigns | None:
        if self.status != "running":
            return self.result
        return self._finish(status="stopped", noise_filter=noise_filter, calculator=calculator)

    def snapshot(self) -> MeasurementState:
        completed_at = self.completed_at
        end_time = completed_at or datetime.now(UTC)
        elapsed_seconds = max(0.0, (end_time - self.started_at).total_seconds())
        progress = 0.0
        if self.duration_seconds is not None:
            progress = min(1.0, elapsed_seconds / self.duration_seconds)

        return MeasurementState(
            id=self.id,
            device_id=self.device_id,
            status=self.status,
            started_at=self.started_at,
            completed_at=completed_at,
            duration_seconds=self.duration_seconds,
            elapsed_seconds=round(elapsed_seconds, 2),
            progress=round(progress, 3),
            samples_collected=len(self._ir),
            result=self.result,
            reason=self.reason,
        )

    def initial_recording_values(self, metadata: RecordingMetadata) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_name": metadata.user_name,
            "user_id": metadata.user_id,
            "project_name": metadata.project_name,
            "project_id": metadata.project_id,
            "started_at": self.started_at,
            "finished_at": None,
            "duration_ms": None,
            "bpm": None,
            "spo2": None,
            "status": self.status,
            "signal_type": self._signal_type(),
            "sample_rate": self._fs,
            "created_at": self.started_at,
            "updated_at": self.started_at,
            "signal_quality": None,
            "sensor_temp": self._temperature,
            "device_id": self.device_id,
            "perfusion_index": None,
            "ratio": None,
            "sensor_confidence": None,
            "peak_count": None,
        }

    def recording_update(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "status": self.status,
            "signal_type": self._signal_type(),
            "sample_rate": self._calculation_fs() or self._fs,
            "updated_at": datetime.now(UTC),
            "sensor_temp": self._temperature,
            "device_id": self.device_id,
        }

        if self.completed_at is not None:
            values["finished_at"] = self.completed_at
            values["duration_ms"] = int(round((self.completed_at - self.started_at).total_seconds() * 1000))

        if self.result is not None:
            quality = self.result.signal_quality
            values.update(
                {
                    "bpm": self.result.bpm,
                    "spo2": self.result.spo2,
                    "signal_quality": quality.model_dump(mode="json"),
                    "perfusion_index": quality.perfusion_index,
                    "ratio": self.result.ratio,
                    "sensor_confidence": self.result.sensor_confidence,
                    "peak_count": quality.peak_count,
                }
            )

        return values

    def _samples_for_buffer_duration(self, samples: list[PPGSample]) -> list[PPGSample]:
        if self.duration_seconds is None or self._fs is None:
            return samples
        target_samples = max(1, int(ceil(self.duration_seconds * self._fs)))
        remaining = max(0, target_samples - len(self._ir))
        return samples[:remaining]

    def _append_samples(self, samples: list[PPGSample]) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []
        for sample in samples:
            raw_data = {"ir": float(sample.ir), "r": float(sample.red)}
            self._ir.append(raw_data["ir"])
            self._red.append(raw_data["r"])
            rows.append(
                {
                    "sample_index": self._next_sample_index,
                    "raw_data": raw_data,
                    "created_at": now,
                }
            )
            self._next_sample_index += 1
        return rows

    def _finish(
        self,
        *,
        status: str,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> VitalSigns | None:
        self.completed_at = datetime.now(UTC)
        self.status = status

        if self._fs is None or not self._ir:
            return None

        calculation_fs = self._calculation_fs() or self._fs
        raw_ir = np.asarray(self._ir, dtype=float)
        raw_red = np.asarray(self._red, dtype=float)
        filtered_ir, filtered_red = noise_filter.filter_pair(raw_ir, raw_red, calculation_fs)
        result = calculator.calculate(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            filtered_red=filtered_red,
            fs=calculation_fs,
        )
        self.result = VitalSigns(
            device_id=self.device_id or "",
            timestamp=self.completed_at,
            fs=calculation_fs,
            temperature=self._temperature,
            bpm=result.bpm,
            spo2=result.spo2,
            ratio=result.ratio,
            sensor_confidence=result.sensor_confidence,
            signal_quality=result.quality,
            waveform_morphology=result.waveform_morphology,
        )
        return self.result

    def _fail(self, reason: str) -> None:
        self.completed_at = datetime.now(UTC)
        self.status = "failed"
        self.reason = reason

    def _duration_ready(self) -> bool:
        return self._buffer_duration_elapsed()

    def _wall_clock_duration_elapsed(self) -> bool:
        if self.duration_seconds is None:
            return False
        return (datetime.now(UTC) - self.started_at).total_seconds() >= self.duration_seconds

    def _buffer_duration_elapsed(self) -> bool:
        if self.duration_seconds is None or self._fs is None:
            return False
        return len(self._ir) >= max(1, int(ceil(self.duration_seconds * self._fs)))

    def _calculation_fs(self) -> float | None:
        if self._fs is not None:
            return self._fs

        effective_fs = self._effective_fs()
        if self.duration_seconds is None:
            return effective_fs

        end_time = self.completed_at or datetime.now(UTC)
        elapsed_seconds = max(0.0, (end_time - self.started_at).total_seconds())
        if elapsed_seconds >= self.duration_seconds * 0.8:
            return effective_fs
        return None

    def _effective_fs(self) -> float | None:
        end_time = self.completed_at or datetime.now(UTC)
        elapsed_seconds = max(0.0, (end_time - self.started_at).total_seconds())
        if elapsed_seconds <= 0 or not self._ir:
            return None
        return round(len(self._ir) / elapsed_seconds, 3)

    def _signal_type(self) -> str:
        if self._ir and self._red:
            return "IR+R"
        if self._ir:
            return "IR"
        if self._red:
            return "R"
        return "IR+R"


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

        session = MeasurementSession(
            recording_id=str(uuid4()),
            device_id=device_id,
            duration_seconds=duration_seconds,
        )
        self._create_recording(session, metadata)

        operations: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
        results: list[VitalSigns] = []
        with self._lock:
            if device_id is not None:
                for existing in self._sessions.values():
                    if existing.device_id == device_id and existing.status == "running":
                        result = existing.stop(self._filter, self._calculator)
                        operations.append((existing.id, [], existing.recording_update()))
                        if result is not None:
                            results.append(result)
            self._sessions[session.id] = session
            if device_id is not None:
                self._recording_by_device[device_id] = session.id
            snapshot = session.snapshot()

        self._persist_operations(operations)
        for result in results:
            self._print_result(result)
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
            if session is None:
                return None
            was_running = session.status == "running"
            result = session.stop(self._filter, self._calculator)
            snapshot = session.snapshot()
            operation = (session.id, [], session.recording_update())

        self._persist_operations([operation])
        if was_running and result is not None:
            self._print_result(result)
        return snapshot

    def stop_recording_for_device(self, device_id: str) -> MeasurementState | None:
        with self._lock:
            recording_id = self._recording_by_device.get(device_id)
            if recording_id is None:
                return None

        return self.stop_recording(recording_id)

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

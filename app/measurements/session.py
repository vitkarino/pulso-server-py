from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import ceil
from typing import Any

import numpy as np

from app.core.ids import public_device_id
from app.measurements.models import RecordingEvent, RecordingMetadata
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator
from app.schemas.device import DeviceData, PPGSample
from app.schemas.measurements import MeasurementState
from app.schemas.metrics import VitalSigns


@dataclass
class ActiveRecording:
    id: str
    public_id: str
    started_at: datetime
    duration_seconds: float | None
    sample_start_index: int
    sample_end_index: int | None = None
    samples_count: int = 0
    completed_at: datetime | None = None
    status: str = "running"
    result: VitalSigns | None = None
    ir: list[float] = field(default_factory=list)
    red: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class SessionAppendResult:
    result: VitalSigns | None
    sample_rows: list[dict[str, Any]]
    measurement_update: dict[str, Any]
    recording_id: str | None = None
    recording_update: dict[str, Any] | None = None
    events: tuple[RecordingEvent, ...] = ()
    measurement_finished: bool = False


class MeasurementSession:
    def __init__(
        self,
        *,
        measurement_id: str,
        public_measurement_id: str,
        device_id: str | None,
        duration_seconds: float | None,
        metadata: RecordingMetadata,
    ) -> None:
        self.internal_id = measurement_id
        self.public_id = public_measurement_id
        self.device_id = device_id
        self.duration_seconds = duration_seconds
        self.metadata = metadata
        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None
        self.status = "running"
        self.reason: str | None = None
        self.result: VitalSigns | None = None
        self.active_recording: ActiveRecording | None = None
        self._fs: float | None = None
        self._temperature: float | None = None
        self._ir: list[float] = []
        self._red: list[float] = []
        self._next_sample_index = 0

    @property
    def id(self) -> str:
        return self.public_id

    def append(
        self,
        device: DeviceData,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> SessionAppendResult:
        if self.status != "running":
            return SessionAppendResult(
                result=self.result,
                sample_rows=[],
                measurement_update=self.measurement_update(),
            )

        if device.measurement_id is not None and device.measurement_id not in {self.internal_id, self.public_id}:
            return SessionAppendResult(result=None, sample_rows=[], measurement_update={})

        if self.device_id is None:
            self.device_id = device.id
        elif self.device_id != device.id:
            return SessionAppendResult(result=None, sample_rows=[], measurement_update={})

        if self._fs is None:
            self._fs = device.fs
        elif self._fs != device.fs:
            self._fail("sampling frequency changed during measurement")
            recording_update, event = self._stop_active_recording(
                status="failed",
                noise_filter=noise_filter,
                calculator=calculator,
            )
            return SessionAppendResult(
                result=None,
                sample_rows=[],
                measurement_update=self.measurement_update(),
                recording_id=event.recording_id if event is not None else None,
                recording_update=recording_update,
                events=(event,) if event is not None else (),
                measurement_finished=True,
            )

        self._temperature = device.temp
        sample_rows = self._append_samples(self._samples_for_buffer_duration(device.samples))
        recording_update = None
        recording_id = None
        events: list[RecordingEvent] = []

        if self.active_recording is not None:
            recording_id = self.active_recording.id
            recording_update = self.recording_update(self.active_recording)
            if self._recording_duration_ready(self.active_recording):
                recording_update, event = self._stop_active_recording(
                    status="completed",
                    noise_filter=noise_filter,
                    calculator=calculator,
                )
                if event is not None:
                    events.append(event)

        result: VitalSigns | None = None
        measurement_finished = False
        if self._duration_ready():
            result = self._finish_measurement(
                status="completed",
                noise_filter=noise_filter,
                calculator=calculator,
            )
            measurement_finished = True
            active_update, event = self._stop_active_recording(
                status="completed",
                noise_filter=noise_filter,
                calculator=calculator,
            )
            if active_update is not None:
                recording_update = active_update
            if event is not None:
                events.append(event)

        return SessionAppendResult(
            result=result,
            sample_rows=sample_rows,
            measurement_update=self.measurement_update(),
            recording_id=recording_id,
            recording_update=recording_update,
            events=tuple(events),
            measurement_finished=measurement_finished,
        )

    def start_recording(
        self,
        *,
        recording_id: str,
        public_recording_id: str,
        duration_seconds: float | None,
    ) -> ActiveRecording:
        if self.status != "running":
            raise ValueError("measurement is not active")
        if self.active_recording is not None and self.active_recording.status == "running":
            raise ValueError("recording is already running")
        self.active_recording = ActiveRecording(
            id=recording_id,
            public_id=public_recording_id,
            started_at=datetime.now(UTC),
            duration_seconds=duration_seconds,
            sample_start_index=self._next_sample_index,
        )
        return self.active_recording

    def stop_recording(
        self,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> tuple[dict[str, Any] | None, RecordingEvent | None]:
        return self._stop_active_recording(
            status="completed",
            noise_filter=noise_filter,
            calculator=calculator,
        )

    def stop_measurement(
        self,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> tuple[VitalSigns | None, dict[str, Any] | None, RecordingEvent | None]:
        if self.status != "running":
            return self.result, None, None
        result = self._finish_measurement(status="cancelled", noise_filter=noise_filter, calculator=calculator)
        recording_update, event = self._stop_active_recording(
            status="cancelled",
            noise_filter=noise_filter,
            calculator=calculator,
        )
        return result, recording_update, event

    def snapshot(self) -> MeasurementState:
        completed_at = self.completed_at
        end_time = completed_at or datetime.now(UTC)
        elapsed_seconds = max(0.0, (end_time - self.started_at).total_seconds())
        progress = 0.0
        if self.duration_seconds is not None:
            progress = min(1.0, elapsed_seconds / self.duration_seconds)

        return MeasurementState(
            id=self.public_id,
            user_id=self.metadata.user_id,
            project_id=self.metadata.project_id,
            device_id=public_device_id(self.device_id),
            active_recording_id=self.active_recording.public_id
            if self.active_recording is not None and self.active_recording.status == "running"
            else None,
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

    def initial_measurement_values(self) -> dict[str, Any]:
        return {
            "id": self.internal_id,
            "public_id": self.public_id,
            "user_id": self.metadata.user_id,
            "project_id": self.metadata.project_id,
            "device_id": self.device_id,
            "active_recording_id": None,
            "started_at": self.started_at,
            "finished_at": None,
            "duration_ms": None,
            "status": self.status,
            "signal_type": self._signal_type(),
            "sample_rate": self._fs,
            "sensor_temp": self._temperature,
            "bpm": None,
            "spo2": None,
            "ratio": None,
            "signal_quality": None,
            "peak_count": None,
            "created_at": self.started_at,
            "updated_at": self.started_at,
        }

    def initial_recording_values(self, recording: ActiveRecording) -> dict[str, Any]:
        return {
            "id": recording.id,
            "public_id": recording.public_id,
            "measurement_id": self.internal_id,
            "quality_analysis_id": None,
            "use_for_ml_training": False,
            "user_name": self.metadata.user_name,
            "user_id": self.metadata.user_id,
            "project_name": self.metadata.project_name,
            "project_id": self.metadata.project_id,
            "started_at": recording.started_at,
            "finished_at": None,
            "duration_ms": None,
            "bpm": None,
            "spo2": None,
            "status": recording.status,
            "signal_type": self._signal_type(),
            "sample_rate": self._fs,
            "created_at": recording.started_at,
            "updated_at": recording.started_at,
            "signal_quality": None,
            "sensor_temp": self._temperature,
            "device_id": self.device_id,
            "perfusion_index": None,
            "ratio": None,
            "peak_count": None,
            "sample_start_index": recording.sample_start_index,
            "sample_end_index": None,
            "samples_count": 0,
        }

    def measurement_update(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "status": self.status,
            "signal_type": self._signal_type(),
            "sample_rate": self._calculation_fs() or self._fs,
            "updated_at": datetime.now(UTC),
            "sensor_temp": self._temperature,
            "device_id": self.device_id,
            "active_recording_id": self.active_recording.id
            if self.active_recording is not None and self.active_recording.status == "running"
            else None,
        }

        if self.completed_at is not None:
            values["finished_at"] = self.completed_at
            values["duration_ms"] = int(round((self.completed_at - self.started_at).total_seconds() * 1000))

        if self.result is not None:
            values.update(_metric_values(self.result))

        return values

    def recording_update(self, recording: ActiveRecording) -> dict[str, Any]:
        values: dict[str, Any] = {
            "status": recording.status,
            "signal_type": self._signal_type(),
            "sample_rate": self._calculation_fs() or self._fs,
            "updated_at": datetime.now(UTC),
            "sensor_temp": self._temperature,
            "device_id": self.device_id,
            "sample_start_index": recording.sample_start_index,
            "sample_end_index": recording.sample_end_index,
            "samples_count": recording.samples_count,
        }

        if recording.completed_at is not None:
            values["finished_at"] = recording.completed_at
            values["duration_ms"] = int(round((recording.completed_at - recording.started_at).total_seconds() * 1000))

        if recording.result is not None:
            values.update(_metric_values(recording.result))

        return values

    def _samples_for_buffer_duration(self, samples: list[PPGSample]) -> list[PPGSample]:
        if self.duration_seconds is None or self._fs is None:
            return samples
        target_samples = self._target_sample_count()
        remaining = max(0, target_samples - len(self._ir))
        return samples[:remaining]

    def _append_samples(self, samples: list[PPGSample]) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        rows: list[dict[str, Any]] = []
        recording = self.active_recording if self.active_recording and self.active_recording.status == "running" else None
        for sample in samples:
            sample_index = sample.index if sample.index is not None else self._next_sample_index
            raw_data = {"ir": float(sample.ir), "red": float(sample.red)}
            self._ir.append(raw_data["ir"])
            self._red.append(raw_data["red"])
            if recording is not None:
                recording.ir.append(raw_data["ir"])
                recording.red.append(raw_data["red"])
                recording.sample_end_index = sample_index
                recording.samples_count += 1
                rows.append(
                    {
                        "sample_index": sample_index,
                        "raw_data": raw_data,
                        "created_at": now,
                    }
                )
            self._next_sample_index = max(self._next_sample_index, sample_index + 1)
        return rows

    def _finish_measurement(
        self,
        *,
        status: str,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> VitalSigns | None:
        self.completed_at = datetime.now(UTC)
        self.status = status

        self.result = self._calculate_result(
            ir=self._ir,
            red=self._red,
            timestamp=self.completed_at,
            noise_filter=noise_filter,
            calculator=calculator,
        )
        return self.result

    def _stop_active_recording(
        self,
        *,
        status: str,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> tuple[dict[str, Any] | None, RecordingEvent | None]:
        recording = self.active_recording
        if recording is None or recording.status != "running":
            return None, None

        recording.completed_at = datetime.now(UTC)
        recording.status = status
        if recording.sample_end_index is None and recording.samples_count > 0:
            recording.sample_end_index = recording.sample_start_index + recording.samples_count - 1
        recording.result = self._calculate_result(
            ir=recording.ir,
            red=recording.red,
            timestamp=recording.completed_at,
            noise_filter=noise_filter,
            calculator=calculator,
        )
        event = RecordingEvent(
            type="recording_stopped",
            measurement_id=self.public_id,
            recording_id=recording.public_id,
            sample_start_index=recording.sample_start_index,
            sample_end_index=recording.sample_end_index,
            samples_count=recording.samples_count,
        )
        values = self.recording_update(recording)
        self.active_recording = None
        return values, event

    def _calculate_result(
        self,
        *,
        ir: list[float],
        red: list[float],
        timestamp: datetime,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> VitalSigns | None:
        if self._fs is None or not ir or not red:
            return None

        calculation_fs = self._calculation_fs() or self._fs
        raw_ir = np.asarray(ir, dtype=float)
        raw_red = np.asarray(red, dtype=float)
        filtered_ir, filtered_red = noise_filter.filter_pair(raw_ir, raw_red, calculation_fs)
        result = calculator.calculate(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            filtered_red=filtered_red,
            fs=calculation_fs,
        )
        return VitalSigns(
            device_id=self.device_id or "",
            timestamp=timestamp,
            fs=calculation_fs,
            temperature=self._temperature,
            bpm=result.bpm,
            spo2=result.spo2,
            ratio=result.ratio,
            signal_quality=result.quality,
        )

    def _fail(self, reason: str) -> None:
        self.completed_at = datetime.now(UTC)
        self.status = "failed"
        self.reason = reason

    def _duration_ready(self) -> bool:
        return self._buffer_duration_elapsed()

    def _recording_duration_ready(self, recording: ActiveRecording) -> bool:
        if recording.duration_seconds is None or self._fs is None:
            return False
        target = max(1, int(ceil(recording.duration_seconds * self._fs - 1e-9)))
        return recording.samples_count >= target

    def _buffer_duration_elapsed(self) -> bool:
        if self.duration_seconds is None or self._fs is None:
            return False
        return len(self._ir) >= self._target_sample_count()

    def _target_sample_count(self) -> int:
        if self.duration_seconds is None or self._fs is None:
            return 0
        return max(1, int(ceil(self.duration_seconds * self._fs - 1e-9)))

    def _calculation_fs(self) -> float | None:
        return self._fs

    def _signal_type(self) -> str:
        if self._ir and self._red:
            return "IR+R"
        if self._ir:
            return "IR"
        if self._red:
            return "R"
        return "IR+R"


def _metric_values(result: VitalSigns) -> dict[str, Any]:
    quality = result.signal_quality
    return {
        "bpm": result.bpm,
        "spo2": result.spo2,
        "signal_quality": quality.model_dump(mode="json"),
        "perfusion_index": quality.perfusion_index,
        "ratio": result.ratio,
        "peak_count": quality.peak_count,
    }

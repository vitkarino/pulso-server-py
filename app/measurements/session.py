from __future__ import annotations

from datetime import UTC, datetime
from math import ceil
from typing import Any

import numpy as np

from app.measurements.models import RecordingMetadata
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator
from app.schemas.device import DeviceData, PPGSample
from app.schemas.measurements import MeasurementState
from app.schemas.metrics import VitalSigns


class MeasurementSession:
    def __init__(
        self,
        *,
        recording_id: str,
        device_id: str | None,
        duration_seconds: float | None,
        metadata: RecordingMetadata,
    ) -> None:
        self.id = recording_id
        self.device_id = device_id
        self.duration_seconds = duration_seconds
        self.metadata = metadata
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
            user_id=self.metadata.user_id,
            project_id=self.metadata.project_id,
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
        target_samples = self._target_sample_count()
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
        return len(self._ir) >= self._target_sample_count()

    def _target_sample_count(self) -> int:
        if self.duration_seconds is None or self._fs is None:
            return 0
        return max(1, int(ceil(self.duration_seconds * self._fs - 1e-9)))

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

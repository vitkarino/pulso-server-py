from datetime import UTC, datetime
import json
from math import ceil
from threading import RLock

import numpy as np

from app.config import AppConfig
from app.models import DeviceData, MeasurementState, PPGSample, VitalSigns
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator


class MeasurementSession:
    def __init__(self, device_id: str, duration_seconds: float) -> None:
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

    def append(
        self,
        samples: list[PPGSample],
        fs: float,
        temperature: float | None,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> VitalSigns | None:
        if self.status != "running":
            return self.result

        if self._fs is None:
            self._fs = fs
        elif self._fs != fs:
            self._fail("sampling frequency changed during measurement")
            return None

        self._temperature = temperature
        target_samples = self._target_samples()
        remaining = max(0, target_samples - len(self._ir))
        for sample in samples[:remaining]:
            self._ir.append(float(sample.ir))
            self._red.append(float(sample.red))

        if len(self._ir) < target_samples:
            return None

        return self._complete(noise_filter, calculator)

    def snapshot(self) -> MeasurementState:
        completed_at = self.completed_at
        end_time = completed_at or datetime.now(UTC)
        elapsed_seconds = max(0.0, (end_time - self.started_at).total_seconds())
        progress = 0.0
        if self._fs is not None:
            progress = min(1.0, len(self._ir) / self._target_samples())

        return MeasurementState(
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

    def _target_samples(self) -> int:
        if self._fs is None:
            return 1
        return max(1, int(ceil(self.duration_seconds * self._fs)))

    def _complete(
        self,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> VitalSigns:
        assert self._fs is not None
        raw_ir = np.asarray(self._ir, dtype=float)
        raw_red = np.asarray(self._red, dtype=float)
        filtered_ir, filtered_red = noise_filter.filter_pair(raw_ir, raw_red, self._fs)
        result = calculator.calculate(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            filtered_red=filtered_red,
            fs=self._fs,
        )
        self.completed_at = datetime.now(UTC)
        self.status = "completed"
        self.result = VitalSigns(
            device_id=self.device_id,
            timestamp=self.completed_at,
            fs=self._fs,
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


class MeasurementManager:
    def __init__(
        self,
        config: AppConfig,
        noise_filter: PPGNoiseFilter,
        calculator: VitalSignsCalculator,
    ) -> None:
        self._config = config
        self._filter = noise_filter
        self._calculator = calculator
        self._lock = RLock()
        self._sessions: dict[str, MeasurementSession] = {}

    def start(self, device_id: str, duration_seconds: float | None = None) -> MeasurementState:
        duration = duration_seconds or self._config.measurement_duration_seconds
        if duration < self._config.min_window_seconds:
            raise ValueError(
                f"measurement duration must be at least {self._config.min_window_seconds} seconds"
            )

        with self._lock:
            session = MeasurementSession(device_id=device_id, duration_seconds=duration)
            self._sessions[device_id] = session
            return session.snapshot()

    def ingest(self, device: DeviceData) -> VitalSigns | None:
        with self._lock:
            session = self._sessions.get(device.id)
            if session is None or session.status != "running":
                return None
            result = session.append(
                samples=device.samples,
                fs=device.fs,
                temperature=device.temp,
                noise_filter=self._filter,
                calculator=self._calculator,
            )

        if result is not None:
            self._print_result(result)
        return result

    def get(self, device_id: str) -> MeasurementState | None:
        with self._lock:
            session = self._sessions.get(device_id)
            if session is None:
                return None
            return session.snapshot()

    def all(self) -> dict[str, MeasurementState]:
        with self._lock:
            return {device_id: session.snapshot() for device_id, session in self._sessions.items()}

    @staticmethod
    def _print_result(result: VitalSigns) -> None:
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False), flush=True)

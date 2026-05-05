from dataclasses import dataclass
from datetime import UTC, datetime
import json

import numpy as np
from pydantic import ValidationError

from app.core.config import AppConfig
from app.measurements import CompletedMeasurement, MeasurementManager, RecordingEvent, RecordingMetadata
from app.processing.buffer import DeviceBufferRegistry
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator
from app.processing.quality import QualityAnalysisResult, QualityAnalyzer
from app.realtime.store import MetricsStore
from app.schemas.device import DevicePayload
from app.schemas.metrics import VitalSigns
from app.storage.recording_repository import RecordingRepository


@dataclass(frozen=True)
class ProcessedDeviceMessage:
    metrics: VitalSigns
    completed_measurements: list[CompletedMeasurement]
    recording_events: list[RecordingEvent]
    processed_samples: list[dict[str, float]]


class PPGProcessingService:
    def __init__(
        self,
        config: AppConfig,
        store: MetricsStore,
        recording_repository: RecordingRepository | None = None,
    ) -> None:
        self._config = config
        self._buffers = DeviceBufferRegistry(config)
        self._filter = PPGNoiseFilter(config)
        self._calculator = VitalSignsCalculator(config)
        self._quality_analyzer = QualityAnalyzer(config)
        self._recording_repository = recording_repository
        self._measurements = MeasurementManager(
            config,
            self._filter,
            self._calculator,
            recording_repository,
        )
        self._store = store

    @property
    def database_enabled(self) -> bool:
        return self._recording_repository is not None and self._recording_repository.enabled

    def prepare_database(self) -> None:
        if self._recording_repository is not None and self._recording_repository.enabled:
            self._recording_repository.create_schema()

    def process_json(self, raw_message: str | bytes) -> VitalSigns:
        return self._process_device_message(raw_message, ingest_recordings=False).metrics

    def process_json_with_recordings(
        self,
        raw_message: str | bytes,
    ) -> tuple[VitalSigns, list[CompletedMeasurement]]:
        processed = self._process_device_message(raw_message, ingest_recordings=True)
        return processed.metrics, processed.completed_measurements

    def process_json_with_recordings_and_signal(self, raw_message: str | bytes) -> ProcessedDeviceMessage:
        return self._process_device_message(raw_message, ingest_recordings=True)

    def _process_device_message(
        self,
        raw_message: str | bytes,
        *,
        ingest_recordings: bool,
    ) -> ProcessedDeviceMessage:
        try:
            payload = DevicePayload.model_validate_json(raw_message)
        except ValidationError:
            raise

        device = payload.device
        window = self._buffers.window_for(device.id, device.samples, device.fs)
        filtered_ir, filtered_red = self._filter.filter_pair(window.ir, window.red, window.fs)
        result = self._calculator.calculate(
            raw_ir=window.ir,
            raw_red=window.red,
            filtered_ir=filtered_ir,
            filtered_red=filtered_red,
            fs=window.fs,
        )

        metrics = VitalSigns(
            device_id=device.id,
            timestamp=datetime.now(UTC),
            fs=device.fs,
            temperature=device.temp,
            bpm=result.bpm,
            spo2=result.spo2,
            ratio=result.ratio,
            signal_quality=result.quality,
        )
        self._store.update(metrics)
        if self._config.print_live_metrics:
            self._print_metrics(metrics)
        if ingest_recordings:
            ingest_result = self._measurements.ingest_recordings(device)
        else:
            self._measurements.ingest(device)
            ingest_result = None

        return ProcessedDeviceMessage(
            metrics=metrics,
            completed_measurements=ingest_result.completed_measurements if ingest_result is not None else [],
            recording_events=ingest_result.recording_events if ingest_result is not None else [],
            processed_samples=self._processed_latest_samples(
                filtered_ir=filtered_ir,
                filtered_red=filtered_red,
                batch_size=len(device.samples),
            ),
        )

    @staticmethod
    def _processed_latest_samples(
        *,
        filtered_ir: np.ndarray,
        filtered_red: np.ndarray,
        batch_size: int,
    ) -> list[dict[str, float]]:
        sample_count = min(batch_size, filtered_ir.size, filtered_red.size)
        if sample_count <= 0:
            return []

        ir_tail = filtered_ir[-sample_count:]
        red_tail = filtered_red[-sample_count:]
        return PPGProcessingService._sample_dicts_from_arrays(ir_tail, red_tail)

    @staticmethod
    def _sample_dicts_from_arrays(
        ir_values: np.ndarray | list[float],
        red_values: np.ndarray | list[float],
    ) -> list[dict[str, float]]:
        return [
            {
                "ir": float(ir),
                "r": float(red),
            }
            for ir, red in zip(ir_values, red_values, strict=True)
        ]

    def start_measurement(self, device_id: str):
        if device_id is not None:
            self._buffers.reset(device_id)
        return self._measurements.start(device_id)

    def start_live_measurement(
        self,
        *,
        metadata: RecordingMetadata,
        device_id: str,
    ):
        self._buffers.reset(device_id)
        return self._measurements.start_measurement(
            metadata=metadata,
            device_id=device_id,
        )

    def start_recording(
        self,
        *,
        duration_seconds: float | None,
        metadata: RecordingMetadata,
        device_id: str | None = None,
    ):
        if device_id is not None:
            self._buffers.reset(device_id)
        measurement = self._measurements.start_measurement(
            metadata=metadata,
            device_id=device_id,
        )
        if measurement.id is not None:
            self._measurements.start_recording_for_measurement(
                measurement.id,
                duration_seconds=duration_seconds,
            )
        return self._measurements.get_recording_state(measurement.id or "") or measurement

    def start_recording_for_measurement(self, measurement_id: str, duration_seconds: float | None = None):
        return self._measurements.start_recording_for_measurement(
            measurement_id,
            duration_seconds=duration_seconds,
        )

    def stop_measurement(self, measurement_id: str):
        return self._measurements.stop_measurement(measurement_id)

    def stop_recording(self, measurement_id: str):
        return self._measurements.stop_recording(measurement_id)

    def stop_recording_for_measurement(self, measurement_id: str):
        return self._measurements.stop_recording_for_measurement(measurement_id)

    def get_recording_state(self, recording_id: str):
        return self._measurements.get_recording_state(recording_id)

    def stop_recording_for_device(self, device_id: str):
        return self._measurements.stop_recording_for_device(device_id)

    def forget_device(self, device_id: str) -> None:
        self._buffers.reset(device_id)

    def delete_recording(self, recording_id: str):
        return self._measurements.delete_recording(recording_id)

    def stop_all_recordings(self):
        return self._measurements.stop_all()

    def get_measurement(self, device_id: str):
        return self._measurements.get(device_id)

    def get_measurements(self):
        return self._measurements.all()

    def get_measurement_record(self, measurement_id: str):
        repository = self._require_recording_repository()
        return repository.get_measurement(measurement_id)

    def list_measurements(
        self,
        *,
        limit: int | None,
        offset: int,
        date_from=None,
        date_to=None,
        device_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
        status: str | None = None,
    ):
        repository = self._require_recording_repository()
        return repository.list_measurements(
            limit=limit,
            offset=offset,
            date_from=date_from,
            date_to=date_to,
            device_id=device_id,
            user_id=user_id,
            project_id=project_id,
            status=status,
        )

    def get_recording(self, recording_id: str):
        repository = self._require_recording_repository()
        return repository.get_recording(recording_id)

    def list_recordings(
        self,
        *,
        limit: int | None,
        offset: int,
        date_from=None,
        date_to=None,
        device_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
        status: str | None = None,
    ):
        repository = self._require_recording_repository()
        return repository.list_recordings(
            limit=limit,
            offset=offset,
            date_from=date_from,
            date_to=date_to,
            device_id=device_id,
            user_id=user_id,
            project_id=project_id,
            status=status,
        )

    def get_recording_samples(self, recording_id: str, *, limit: int | None, offset: int):
        repository = self._require_recording_repository()
        return repository.list_recording_samples(recording_id, limit=limit, offset=offset)

    def get_recording_processed_samples(
        self,
        recording_id: str,
        *,
        limit: int | None,
        offset: int,
    ) -> list[dict[str, float]]:
        repository = self._require_recording_repository()
        recording = repository.get_recording(recording_id)
        if recording is None:
            return []

        rows = repository.list_recording_samples(recording_id, limit=None, offset=0)
        if not rows:
            return []

        raw_ir: list[float] = []
        raw_red: list[float] = []
        for row in rows:
            raw_data = row.get("raw_data")
            if not isinstance(raw_data, dict):
                continue
            ir = raw_data.get("ir")
            red = raw_data.get("r", raw_data.get("red"))
            if ir is None or red is None:
                continue
            raw_ir.append(float(ir))
            raw_red.append(float(red))

        if not raw_ir or not raw_red:
            return []

        sample_rate = recording.get("sample_rate")
        if sample_rate is None:
            processed = self._sample_dicts_from_arrays(raw_ir, raw_red)
        else:
            filtered_ir, filtered_red = self._filter.filter_pair(
                np.asarray(raw_ir, dtype=float),
                np.asarray(raw_red, dtype=float),
                float(sample_rate),
            )
            processed = self._sample_dicts_from_arrays(filtered_ir, filtered_red)

        if limit is None:
            return processed[offset:]
        return processed[offset : offset + limit]

    def get_recording_samples_for_api(
        self,
        recording_id: str,
        *,
        limit: int | None,
        offset: int,
    ) -> list[dict[str, float | int | None]]:
        repository = self._require_recording_repository()
        recording = repository.get_recording(recording_id)
        if recording is None:
            return []
        rows = repository.list_recording_samples(recording_id, limit=None, offset=0)
        if not rows:
            return []

        raw_ir, raw_red = self._raw_arrays_from_rows(rows)
        sample_rate = recording.get("sample_rate")
        if sample_rate is not None and raw_ir.size and raw_red.size:
            filtered_ir, filtered_red = self._filter.filter_pair(raw_ir, raw_red, float(sample_rate))
        else:
            filtered_ir, filtered_red = raw_ir, raw_red

        output: list[dict[str, float | int | None]] = []
        for index, row in enumerate(rows):
            raw_data = row.get("raw_data") if isinstance(row, dict) else None
            raw = raw_data if isinstance(raw_data, dict) else {}
            sample_index = int(row["sample_index"])
            t_ms = None
            if sample_rate:
                t_ms = round(sample_index / float(sample_rate) * 1000.0, 3)
            output.append(
                {
                    "index": sample_index,
                    "t_ms": t_ms,
                    "ir": raw.get("ir"),
                    "red": raw.get("red", raw.get("r")),
                    "ir_filtered": float(filtered_ir[index]) if index < filtered_ir.size else None,
                    "red_filtered": float(filtered_red[index]) if index < filtered_red.size else None,
                }
            )
        if limit is None:
            return output[offset:]
        return output[offset : offset + limit]

    def analyze_recording_quality(self, recording_id: str) -> QualityAnalysisResult | dict[str, object]:
        repository = self._require_recording_repository()
        recording = repository.get_recording(recording_id)
        if recording is None:
            raise KeyError("recording not found")
        rows = repository.list_recording_samples(recording_id, limit=None, offset=0)
        raw_ir, raw_red = self._raw_arrays_from_rows(rows)
        sample_rate = recording.get("sample_rate")
        if sample_rate is None and recording.get("samples_count") and recording.get("duration_ms"):
            sample_rate = float(recording["samples_count"]) / (float(recording["duration_ms"]) / 1000.0)
        if sample_rate is None:
            raise ValueError("recording sample rate is unknown")
        if raw_ir.size == 0 or raw_red.size == 0:
            raise ValueError("recording has no usable samples")
        filtered_ir, _filtered_red = self._filter.filter_pair(raw_ir, raw_red, float(sample_rate))
        result = self._quality_analyzer.analyze(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            fs=float(sample_rate),
        )
        return repository.upsert_quality_analysis(
            {
                "id": result.id,
                "public_id": result.public_id,
                "recording_id": str(recording["id"]),
                "timestamp": result.timestamp,
                "model": result.model,
                "quality_result": result.quality_result,
                "features": result.features,
                "created_at": result.timestamp,
                "updated_at": result.timestamp,
            }
        )

    def get_quality_analysis(self, recording_id: str):
        repository = self._require_recording_repository()
        return repository.get_quality_analysis_for_recording(recording_id)

    def _require_recording_repository(self) -> RecordingRepository:
        if self._recording_repository is None or not self._recording_repository.enabled:
            raise RuntimeError("DATABASE_URL is not configured")
        return self._recording_repository

    @staticmethod
    def _raw_arrays_from_rows(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
        raw_ir: list[float] = []
        raw_red: list[float] = []
        for row in rows:
            raw_data = row.get("raw_data")
            if not isinstance(raw_data, dict):
                continue
            ir = raw_data.get("ir")
            red = raw_data.get("red", raw_data.get("r"))
            if ir is None or red is None:
                continue
            raw_ir.append(float(ir))
            raw_red.append(float(red))
        return np.asarray(raw_ir, dtype=float), np.asarray(raw_red, dtype=float)

    @staticmethod
    def _print_metrics(metrics: VitalSigns) -> None:
        print(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False), flush=True)

from datetime import UTC, datetime
import json

from pydantic import ValidationError

from app.config import AppConfig
from app.models import DevicePayload, VitalSigns
from app.measurement import CompletedRecording, MeasurementManager, RecordingMetadata
from app.processing.buffer import DeviceBufferRegistry
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator
from app.recording_repository import RecordingRepository
from app.state import MetricsStore


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
            sensor_confidence=result.sensor_confidence,
            signal_quality=result.quality,
            waveform_morphology=result.waveform_morphology,
        )
        self._store.update(metrics)
        if self._config.print_live_metrics:
            self._print_metrics(metrics)
        self._measurements.ingest(device)
        return metrics

    def process_json_with_recordings(
        self,
        raw_message: str | bytes,
    ) -> tuple[VitalSigns, list[CompletedRecording]]:
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
            sensor_confidence=result.sensor_confidence,
            signal_quality=result.quality,
            waveform_morphology=result.waveform_morphology,
        )
        self._store.update(metrics)
        if self._config.print_live_metrics:
            self._print_metrics(metrics)
        completed_recordings = self._measurements.ingest_recordings(device)
        return metrics, completed_recordings

    def start_measurement(self, device_id: str, duration_seconds: float | None = None):
        return self._measurements.start(device_id, duration_seconds)

    def start_recording(
        self,
        *,
        duration_seconds: float | None,
        metadata: RecordingMetadata,
        device_id: str | None = None,
    ):
        if device_id is not None:
            self._buffers.reset(device_id)
        return self._measurements.start_recording(
            duration_seconds=duration_seconds,
            metadata=metadata,
            device_id=device_id,
        )

    def stop_recording(self, recording_id: str):
        return self._measurements.stop_recording(recording_id)

    def stop_recording_for_device(self, device_id: str):
        return self._measurements.stop_recording_for_device(device_id)

    def stop_all_recordings(self):
        return self._measurements.stop_all()

    def get_measurement(self, device_id: str):
        return self._measurements.get(device_id)

    def get_measurements(self):
        return self._measurements.all()

    def get_recording(self, recording_id: str):
        repository = self._require_recording_repository()
        return repository.get_recording(recording_id)

    def list_recordings(self, *, limit: int | None, offset: int, date_from=None, date_to=None):
        repository = self._require_recording_repository()
        return repository.list_recordings(
            limit=limit,
            offset=offset,
            date_from=date_from,
            date_to=date_to,
        )

    def get_recording_samples(self, recording_id: str, *, limit: int | None, offset: int):
        repository = self._require_recording_repository()
        return repository.list_recording_samples(recording_id, limit=limit, offset=offset)

    def _require_recording_repository(self) -> RecordingRepository:
        if self._recording_repository is None or not self._recording_repository.enabled:
            raise RuntimeError("DATABASE_URL is not configured")
        return self._recording_repository

    @staticmethod
    def _print_metrics(metrics: VitalSigns) -> None:
        print(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False), flush=True)

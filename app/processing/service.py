from datetime import UTC, datetime
import json

from pydantic import ValidationError

from app.config import AppConfig
from app.models import DevicePayload, VitalSigns
from app.measurement import MeasurementManager
from app.processing.buffer import DeviceBufferRegistry
from app.processing.filters import PPGNoiseFilter
from app.processing.metrics import VitalSignsCalculator
from app.state import MetricsStore


class PPGProcessingService:
    def __init__(self, config: AppConfig, store: MetricsStore) -> None:
        self._config = config
        self._buffers = DeviceBufferRegistry(config)
        self._filter = PPGNoiseFilter(config)
        self._calculator = VitalSignsCalculator(config)
        self._measurements = MeasurementManager(config, self._filter, self._calculator)
        self._store = store

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
        )
        self._store.update(metrics)
        if self._config.print_live_metrics:
            self._print_metrics(metrics)
        self._measurements.ingest(device)
        return metrics

    def start_measurement(self, device_id: str, duration_seconds: float | None = None):
        return self._measurements.start(device_id, duration_seconds)

    def get_measurement(self, device_id: str):
        return self._measurements.get(device_id)

    def get_measurements(self):
        return self._measurements.all()

    @staticmethod
    def _print_metrics(metrics: VitalSigns) -> None:
        print(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False), flush=True)

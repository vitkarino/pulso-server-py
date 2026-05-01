from dataclasses import dataclass

from app.schemas.metrics import VitalSigns


@dataclass(frozen=True)
class RecordingMetadata:
    user_name: str | None = None
    user_id: str | None = None
    project_name: str | None = None
    project_id: str | None = None


@dataclass(frozen=True)
class CompletedMeasurement:
    measurement_id: str
    device_id: str | None
    result: VitalSigns | None


@dataclass(frozen=True)
class RecordingEvent:
    type: str
    measurement_id: str
    recording_id: str
    sample_start_index: int
    sample_end_index: int | None
    samples_count: int | None

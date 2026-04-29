from dataclasses import dataclass

from app.schemas.metrics import VitalSigns


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

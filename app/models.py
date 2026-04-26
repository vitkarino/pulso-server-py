from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PPGSample(BaseModel):
    ir: float = Field(gt=0)
    red: float = Field(alias="r", gt=0)

    model_config = ConfigDict(populate_by_name=True)


class DeviceData(BaseModel):
    id: str = Field(min_length=1)
    temp: float | None = None
    fs: float = Field(gt=0)
    samples: list[PPGSample] = Field(min_length=1)

    @field_validator("fs")
    @classmethod
    def validate_sampling_frequency(cls, value: float) -> float:
        if value < 5:
            raise ValueError("fs is too low for PPG pulse estimation")
        return value


class DevicePayload(BaseModel):
    device: DeviceData


class SignalQuality(BaseModel):
    level: Literal["warming_up", "no_contact", "low", "medium", "high"]
    samples_in_window: int
    window_seconds: float
    perfusion_index: float | None = None
    peak_count: int = 0
    reason: str | None = None


class VitalSigns(BaseModel):
    device_id: str
    timestamp: datetime
    fs: float
    temperature: float | None = None
    bpm: float | None = None
    spo2: float | None = None
    ratio: float | None = None
    sensor_confidence: float = Field(ge=0.0, le=1.0)
    signal_quality: SignalQuality


class WebSocketAck(BaseModel):
    ok: bool
    metrics: VitalSigns | None = None
    error: str | None = None


class MeasurementState(BaseModel):
    id: str | None = None
    device_id: str | None = None
    status: Literal["running", "completed", "stopped", "failed"]
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None
    elapsed_seconds: float
    progress: float = Field(ge=0.0, le=1.0)
    samples_collected: int
    result: VitalSigns | None = None
    reason: str | None = None

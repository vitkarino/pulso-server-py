from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PPGSample(BaseModel):
    ir: float = Field(gt=0)
    red: float = Field(alias="r", gt=0)

    model_config = ConfigDict(populate_by_name=True)я


class DeviceData(BaseModel):
    id: str = Field(min_length=1)
    recording_id: str | None = Field(default=None, min_length=1)
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
    shape_score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None


class WaveformMorphology(BaseModel):
    enabled: bool = True
    valid_pulse_count: int = 0
    pulse_amplitude: float | None = None
    pulse_duration_ms: float | None = None
    rise_time_ms: float | None = None
    decay_time_ms: float | None = None
    pulse_width_50_ms: float | None = None
    rise_slope: float | None = None
    decay_slope: float | None = None
    area: float | None = None
    symmetry_ratio: float | None = None
    shape_similarity: float | None = None
    amplitude_variability: float | None = None
    duration_variability: float | None = None
    morphology_variability: float | None = None
    shape_score: float | None = Field(default=None, ge=0.0, le=1.0)
    shape_quality: Literal[
        "stable",
        "moderately_stable",
        "unstable",
        "insufficient_pulses",
        "low_amplitude",
        "irregular_shape",
        "invalid_signal",
    ]
    reason: str | None = None
    average_pulse_template: list[float] | None = None


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
    waveform_morphology: WaveformMorphology | None = None


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


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    age: int | None = Field(default=None, ge=0, le=150)
    sex: str | None = Field(default=None, max_length=32)


class UserUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    age: int | None = Field(default=None, ge=0, le=150)
    sex: str | None = Field(default=None, max_length=32)


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = None


class ProjectUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None

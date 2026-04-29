from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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

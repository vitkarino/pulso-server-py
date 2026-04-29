from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.metrics import VitalSigns


class MeasurementState(BaseModel):
    id: str | None = None
    user_id: str | None = None
    project_id: str | None = None
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


class MeasurementStartRequest(BaseModel):
    duration_s: float | None = Field(default=None, gt=0)
    user_name: str | None = Field(default=None, min_length=1)
    user_id: str | None = Field(default=None, min_length=1)
    project_name: str | None = Field(default=None, min_length=1)
    project_id: str | None = Field(default=None, min_length=1)


class UserPatchRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1)
    age: int | None = Field(default=None, ge=0)
    sex: str | None = Field(default=None, min_length=1)


class ProjectPatchRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1)
    description: str | None = None

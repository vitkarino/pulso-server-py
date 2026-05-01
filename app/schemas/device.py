from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.ids import internal_device_id


class PPGSample(BaseModel):
    index: int | None = Field(default=None, ge=0)
    ir: float = Field(gt=0)
    red: float = Field(validation_alias=AliasChoices("red", "r"), gt=0)

    model_config = ConfigDict(populate_by_name=True)


class DeviceData(BaseModel):
    id: str = Field(validation_alias=AliasChoices("id", "device_id"), min_length=1)
    measurement_id: str | None = Field(default=None, min_length=1)
    recording_id: str | None = Field(default=None, min_length=1)
    temp: float | None = Field(default=None, validation_alias=AliasChoices("temp", "sensor_temp_c"))
    fs: float = Field(validation_alias=AliasChoices("fs", "sample_rate_hz"), gt=0)
    samples: list[PPGSample] = Field(min_length=1)

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("id")
    @classmethod
    def normalize_device_id(cls, value: str) -> str:
        normalized = internal_device_id(value)
        if normalized is None or not normalized:
            raise ValueError("device id is required")
        return normalized

    @field_validator("fs")
    @classmethod
    def validate_sampling_frequency(cls, value: float) -> float:
        if value < 5:
            raise ValueError("fs is too low for PPG pulse estimation")
        return value


class DevicePayload(BaseModel):
    device: DeviceData

    @model_validator(mode="before")
    @classmethod
    def normalize_v4_payload(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "device" in value:
            return value
        if value.get("type") != "samples":
            return value
        return {
            "device": {
                "id": value.get("device_id"),
                "measurement_id": value.get("measurement_id"),
                "recording_id": value.get("recording_id"),
                "sample_rate_hz": value.get("sample_rate_hz"),
                "sensor_temp_c": value.get("sensor_temp_c"),
                "samples": value.get("samples"),
            }
        }

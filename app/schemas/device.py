from pydantic import BaseModel, ConfigDict, Field, field_validator


class PPGSample(BaseModel):
    ir: float = Field(gt=0)
    red: float = Field(alias="r", gt=0)

    model_config = ConfigDict(populate_by_name=True)


class DeviceData(BaseModel):
    id: str = Field(min_length=1)
    recording_id: str | None = Field(
        default=None,
        min_length=1,
        validation_alias="measurement_id",
    )
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

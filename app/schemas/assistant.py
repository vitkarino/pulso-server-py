from pydantic import BaseModel, Field, field_validator


class AssistantChatRequest(BaseModel):
    recording_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("recording_id", "message")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

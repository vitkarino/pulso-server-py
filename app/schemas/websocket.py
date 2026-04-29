from pydantic import BaseModel

from app.schemas.metrics import VitalSigns


class WebSocketAck(BaseModel):
    ok: bool
    metrics: VitalSigns | None = None
    error: str | None = None

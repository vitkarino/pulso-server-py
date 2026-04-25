from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.models import WebSocketAck
from app.processing.service import PPGProcessingService


class WebSocketController:
    def __init__(self, service: PPGProcessingService) -> None:
        self._service = service

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_text()
                try:
                    metrics = self._service.process_json(message)
                    ack = WebSocketAck(ok=True, metrics=metrics)
                except ValidationError as exc:
                    ack = WebSocketAck(ok=False, error=exc.errors()[0]["msg"])
                except Exception as exc:
                    ack = WebSocketAck(ok=False, error=str(exc))

                await websocket.send_json(ack.model_dump(mode="json"))
        except WebSocketDisconnect:
            return

from fastapi import FastAPI, HTTPException, Query, WebSocket

from app.config import settings
from app.processing.service import PPGProcessingService
from app.state import metrics_store
from app.websocket_handler import WebSocketController

app = FastAPI(title="Pulso PPG Backend", version="0.1.0")
processing_service = PPGProcessingService(settings, metrics_store)
ws_controller = WebSocketController(processing_service)


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "ws_port": settings.ws_port}


@app.get("/metrics")
def get_metrics() -> dict[str, object]:
    return {"devices": metrics_store.all()}


@app.get("/metrics/{device_id}")
def get_device_metrics(device_id: str) -> object:
    metrics = metrics_store.get(device_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="device metrics not found")
    return metrics


@app.post("/measurements/{device_id}/start")
def start_measurement(
    device_id: str,
    duration_seconds: float | None = Query(default=None, ge=settings.min_window_seconds),
) -> object:
    try:
        return processing_service.start_measurement(device_id, duration_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/measurements")
def get_measurements() -> dict[str, object]:
    return {"measurements": processing_service.get_measurements()}


@app.get("/measurements/{device_id}")
def get_measurement(device_id: str) -> object:
    measurement = processing_service.get_measurement(device_id)
    if measurement is None:
        raise HTTPException(status_code=404, detail="measurement not found")
    return measurement


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_controller.handle(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.ws_port)

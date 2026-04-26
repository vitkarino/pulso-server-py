from datetime import UTC, date, datetime, time
import csv
from io import StringIO
import json
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.measurement import RecordingMetadata
from app.processing.service import PPGProcessingService
from app.recording_repository import RecordingRepository
from app.state import metrics_store
from app.websocket_handler import WebSocketController

app = FastAPI(title="Pulso PPG Backend", version="0.1.0")
recording_repository = RecordingRepository(settings.database_url)
processing_service = PPGProcessingService(settings, metrics_store, recording_repository)
ws_controller = WebSocketController(processing_service)


RECORDING_EXPORT_FIELDS = [
    "id",
    "user_name",
    "user_id",
    "project_name",
    "project_id",
    "started_at",
    "finished_at",
    "duration_ms",
    "bpm",
    "spo2",
    "status",
    "signal_type",
    "sample_rate",
    "created_at",
    "updated_at",
    "signal_quality",
    "sensor_temp",
    "device_id",
    "perfusion_index",
    "ratio",
    "sensor_confidence",
    "peak_count",
]
SAMPLE_EXPORT_FIELDS = ["sample_index", "raw_ir", "raw_red", "raw_data"]


@app.on_event("startup")
def startup() -> None:
    processing_service.prepare_database()


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "ws_port": settings.ws_port}


@app.get("/metrics")
def get_metrics() -> dict[str, object]:
    return {"devices": metrics_store.all()}


@app.get("/devices")
def get_connected_devices() -> dict[str, object]:
    return {"devices": ws_controller.connected_devices()}


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


@app.get("/api/recordings/extract")
def extract_recordings(
    export_format: Literal["json", "csv"] = Query(default="json", alias="format"),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
) -> object:
    _require_database()
    start_bound = _parse_date_bound(date_from, end_of_day=False)
    end_bound = _parse_date_bound(date_to, end_of_day=True)
    recordings = processing_service.list_recordings(
        limit=None,
        offset=0,
        date_from=start_bound,
        date_to=end_bound,
    )
    samples_by_recording = _load_samples_for_recordings(recordings)

    if export_format == "csv":
        return _recordings_csv_response("recordings.csv", recordings, samples_by_recording)

    payload = {
        "recordings": [
            dict(recording, samples=samples_by_recording.get(recording["id"], []))
            for recording in recordings
        ]
    }
    return JSONResponse(content=jsonable_encoder(payload))


@app.get("/api/recordings")
def list_recordings(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
) -> dict[str, object]:
    _require_database()
    return {
        "recordings": processing_service.list_recordings(
            limit=limit,
            offset=offset,
            date_from=_parse_date_bound(date_from, end_of_day=False),
            date_to=_parse_date_bound(date_to, end_of_day=True),
        )
    }


@app.post("/api/recordings/start")
async def start_recording(
    duration: float | None = Query(default=None, gt=0),
    device_id: str = Query(..., min_length=1),
    user_name: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    project_name: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
) -> object:
    _require_database()
    try:
        recording = processing_service.start_recording(
            duration_seconds=duration,
            metadata=RecordingMetadata(
                user_name=user_name,
                user_id=user_id,
                project_name=project_name,
                project_id=project_id,
            ),
            device_id=device_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    command_sent = await ws_controller.send_start(
        device_id=device_id,
        recording_id=recording.id or "",
        duration_seconds=duration,
    )
    if not command_sent:
        processing_service.stop_recording(recording.id or "")
        raise HTTPException(status_code=409, detail="device is not connected")

    return recording


@app.post("/api/recordings/stop-all")
async def stop_all_recordings() -> dict[str, object]:
    _require_database()
    recordings = processing_service.stop_all_recordings()
    stopped_devices: list[str] = []
    for recording in recordings:
        if recording.device_id is not None:
            command_sent = await ws_controller.send_stop(device_id=recording.device_id)
            if command_sent:
                stopped_devices.append(recording.device_id)
    return {"recordings": recordings, "stopped_devices": stopped_devices}


@app.get("/api/recordings/{recording_id}/extract")
def extract_recording(
    recording_id: str,
    export_format: Literal["json", "csv"] = Query(default="json", alias="format"),
) -> object:
    _require_database()
    recording = _get_recording_or_404(recording_id)
    samples = processing_service.get_recording_samples(recording_id, limit=None, offset=0)

    if export_format == "csv":
        return _recordings_csv_response(
            f"recording-{recording_id}.csv",
            [recording],
            {recording_id: samples},
        )

    return JSONResponse(content=jsonable_encoder({"recording": recording, "samples": samples}))


@app.get("/api/recordings/{recording_id}/samples")
def get_recording_samples(
    recording_id: str,
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    _require_database()
    _get_recording_or_404(recording_id)
    return {
        "recording_id": recording_id,
        "samples": processing_service.get_recording_samples(
            recording_id,
            limit=limit,
            offset=offset,
        ),
    }


@app.get("/api/recordings/{recording_id}")
def get_recording(recording_id: str) -> object:
    _require_database()
    return _get_recording_or_404(recording_id)


@app.post("/api/recordings/{recording_id}/stop")
async def stop_recording(recording_id: str) -> object:
    _require_database()
    stopped = processing_service.stop_recording(recording_id)
    if stopped is not None:
        if stopped.device_id is not None:
            await ws_controller.send_stop(device_id=stopped.device_id)
        return stopped

    recording = processing_service.get_recording(recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="recording not found")
    if recording["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="recording is not active in this process",
        )
    return recording


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await ws_controller.handle(websocket)


@app.websocket("/ws/esp32")
async def esp32_websocket_endpoint(websocket: WebSocket) -> None:
    await ws_controller.handle(websocket)


def _require_database() -> None:
    if not processing_service.database_enabled:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")


def _get_recording_or_404(recording_id: str) -> dict[str, object]:
    recording = processing_service.get_recording(recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="recording not found")
    return recording


def _parse_date_bound(raw_value: str | None, *, end_of_day: bool) -> datetime | None:
    if raw_value is None:
        return None

    try:
        parsed_date = date.fromisoformat(raw_value)
    except ValueError:
        parsed_date = None

    if parsed_date is not None:
        bound_time = time.max if end_of_day else time.min
        return datetime.combine(parsed_date, bound_time, tzinfo=UTC)

    try:
        parsed_datetime = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="date filters must use YYYY-MM-DD or ISO datetime format",
        ) from exc

    if parsed_datetime.tzinfo is None:
        return parsed_datetime.replace(tzinfo=UTC)
    return parsed_datetime


def _load_samples_for_recordings(recordings: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    samples_by_recording: dict[str, list[dict[str, object]]] = {}
    for recording in recordings:
        recording_id = str(recording["id"])
        samples_by_recording[recording_id] = processing_service.get_recording_samples(
            recording_id,
            limit=None,
            offset=0,
        )
    return samples_by_recording


def _recordings_csv_response(
    filename: str,
    recordings: list[dict[str, object]],
    samples_by_recording: dict[str, list[dict[str, object]]],
) -> StreamingResponse:
    output = StringIO()
    fieldnames = RECORDING_EXPORT_FIELDS + SAMPLE_EXPORT_FIELDS
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for recording in recordings:
        samples = samples_by_recording.get(str(recording["id"]), [])
        if not samples:
            writer.writerow(_csv_recording_row(recording, None))
            continue

        for sample in samples:
            writer.writerow(_csv_recording_row(recording, sample))

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _csv_recording_row(
    recording: dict[str, object],
    sample: dict[str, object] | None,
) -> dict[str, object]:
    row = {field: _csv_value(recording.get(field)) for field in RECORDING_EXPORT_FIELDS}
    raw_data = sample.get("raw_data") if sample is not None else None
    raw = raw_data if isinstance(raw_data, dict) else {}
    row.update(
        {
            "sample_index": sample.get("sample_index") if sample is not None else None,
            "raw_ir": raw.get("ir"),
            "raw_red": raw.get("r"),
            "raw_data": _csv_value(raw_data),
        }
    )
    return row


def _csv_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.ws_port)

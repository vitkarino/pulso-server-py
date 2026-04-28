from datetime import UTC, date, datetime, time
import csv
from io import StringIO
import json
from time import time_ns
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.measurement import RecordingMetadata
from app.models import MeasurementStartRequest
from app.processing.service import PPGProcessingService
from app.recording_repository import RecordingRepository
from app.state import metrics_store
from app.websocket_handler import WebSocketController

app = FastAPI(
    title="Pulso PPG Backend",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
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
PUBLIC_TO_INTERNAL_STATUS = {
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "stopped",
}
INTERNAL_TO_PUBLIC_STATUS = {
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "stopped": "cancelled",
}


@app.on_event("startup")
def startup() -> None:
    processing_service.prepare_database()


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return _api_error(exc.status_code, _error_code(exc.detail), str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return _api_error(422, "validation_error", exc.errors()[0]["msg"])


@app.get("/api/health")
def api_health() -> JSONResponse:
    return _api_success({"status": "ok", "ws_port": settings.ws_port})


@app.get("/api/devices")
def list_devices(
    connection_status: Literal["connected", "disconnected"] | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    connected = set(ws_controller.connected_devices())
    latest_metrics = metrics_store.all()
    known_device_ids = connected | set(latest_metrics)

    devices = []
    for device_id in sorted(known_device_ids):
        status = "connected" if device_id in connected else "disconnected"
        if connection_status is not None and status != connection_status:
            continue
        devices.append(
            {
                "id": device_id,
                "connection_status": status,
                "metrics": latest_metrics.get(device_id),
            }
        )

    return _api_success(
        {
            "devices": devices[offset : offset + limit],
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/devices/{device_id}/metrics")
def get_api_device_metrics(device_id: str) -> JSONResponse:
    metrics = metrics_store.get(device_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="device metrics not found")
    return _api_success({"metrics": metrics})


@app.get("/api/projects")
def list_projects(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    _require_database()
    return _api_success(
        {
            "projects": recording_repository.list_projects(limit=limit, offset=offset),
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/users")
def list_users(
    project_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    _require_database()
    if project_id is not None:
        _get_project_or_404(project_id)
    return _api_success(
        {
            "users": recording_repository.list_users(
                project_id=project_id,
                limit=limit,
                offset=offset,
            ),
            "limit": limit,
            "offset": offset,
        }
    )


@app.post("/api/devices/{device_id}/measurements")
async def start_api_measurement(
    device_id: str,
    body: MeasurementStartRequest = Body(default_factory=MeasurementStartRequest),
) -> JSONResponse:
    duration_seconds = body.duration_s or settings.measurement_duration_seconds
    try:
        measurement = processing_service.start_recording(
            duration_seconds=duration_seconds,
            metadata=RecordingMetadata(
                user_id=body.user_id,
                project_id=body.project_id,
            ),
            device_id=device_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    command_sent = await ws_controller.send_start(
        device_id=device_id,
        measurement_id=measurement.id or "",
        duration_seconds=duration_seconds,
    )
    if not command_sent:
        processing_service.stop_recording(measurement.id or "")
        raise HTTPException(status_code=409, detail="device is not connected")

    return _api_success({"measurement": _measurement_for_api(measurement)})


@app.post("/api/measurements/{measurement_id}/stop")
async def stop_api_measurement(measurement_id: str) -> JSONResponse:
    stopped = processing_service.stop_recording(measurement_id)
    if stopped is not None:
        if stopped.device_id is not None:
            await ws_controller.send_stop(device_id=stopped.device_id)
        return _api_success({"measurement": _measurement_for_api(stopped)})

    if not processing_service.database_enabled:
        raise HTTPException(status_code=404, detail="measurement not found")

    measurement = processing_service.get_recording(measurement_id)
    if measurement is None:
        raise HTTPException(status_code=404, detail="measurement not found")
    if measurement["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="measurement is not active in this process",
        )
    return _api_success({"measurement": _measurement_for_api(measurement)})


@app.get("/api/measurements")
def list_api_measurements(
    device_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    status: Literal["running", "completed", "failed", "cancelled"] | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    _require_database()
    measurements = processing_service.list_recordings(
        limit=limit,
        offset=offset,
        date_from=_parse_date_bound(date_from, end_of_day=False),
        date_to=_parse_date_bound(date_to, end_of_day=True),
        device_id=device_id,
        user_id=user_id,
        project_id=project_id,
        status=PUBLIC_TO_INTERNAL_STATUS[status] if status is not None else None,
    )
    return _api_success(
        {
            "measurements": [_measurement_for_api(measurement) for measurement in measurements],
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/measurements/{measurement_id}/export")
def export_api_measurement(
    measurement_id: str,
    export_format: Literal["json", "csv"] = Query(default="json", alias="format"),
) -> object:
    _require_database()
    measurement = _get_recording_or_404(measurement_id)
    samples = processing_service.get_recording_samples(measurement_id, limit=None, offset=0)

    if export_format == "csv":
        return _recordings_csv_response(
            f"measurement-{measurement_id}.csv",
            [measurement],
            {measurement_id: samples},
        )

    return _api_success(
        {
            "measurement": _measurement_for_api(measurement),
            "samples": samples,
        }
    )


@app.get("/api/measurements/{measurement_id}/samples")
def get_api_measurement_samples(
    measurement_id: str,
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    _require_database()
    _get_recording_or_404(measurement_id)
    return _api_success(
        {
            "measurement_id": measurement_id,
            "samples": processing_service.get_recording_samples(
                measurement_id,
                limit=limit,
                offset=offset,
            ),
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/measurements/{measurement_id}")
def get_api_measurement(measurement_id: str) -> JSONResponse:
    _require_database()
    return _api_success({"measurement": _measurement_for_api(_get_recording_or_404(measurement_id))})


@app.websocket("/ws/devices/{device_id}")
async def device_websocket_endpoint(websocket: WebSocket, device_id: str) -> None:
    await ws_controller.handle_device(websocket, device_id)


@app.websocket("/ws/measurements/{measurement_id}/stream")
async def measurement_stream_websocket_endpoint(websocket: WebSocket, measurement_id: str) -> None:
    await ws_controller.handle_measurement_stream(websocket, measurement_id)


def _api_success(data: object, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder(
            {
                "data": data,
                "$meta": _meta("success"),
            }
        ),
        status_code=status_code,
    )


def _api_error(http_status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder(
            {
                "data": None,
                "$meta": _meta(
                    "error",
                    error={
                        "http_status": http_status,
                        "code": code,
                        "message": message,
                    },
                ),
            }
        ),
        status_code=http_status,
    )


def _meta(status: Literal["success", "error"], *, error: dict[str, object] | None = None) -> dict[str, object]:
    now = datetime.now(UTC)
    meta: dict[str, object] = {
        "status": status,
        "time": {
            "ts": time_ns(),
            "human": now.strftime("%d/%m/%Y, %H:%M:%S"),
        },
    }
    if error is not None:
        meta["error"] = error
    return meta


def _error_code(detail: object) -> str:
    if not isinstance(detail, str) or not detail:
        return "request_error"
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in detail)
    return "_".join(part for part in normalized.split("_") if part) or "request_error"


def _measurement_for_api(measurement: object) -> object:
    payload = jsonable_encoder(measurement)
    if isinstance(payload, dict):
        _rewrite_measurement_status(payload)
    return payload


def _rewrite_measurement_status(payload: dict[str, object]) -> None:
    status = payload.get("status")
    if isinstance(status, str):
        payload["status"] = INTERNAL_TO_PUBLIC_STATUS.get(status, status)
    if "duration_seconds" in payload and "duration_s" not in payload:
        payload["duration_s"] = payload["duration_seconds"]


def _require_database() -> None:
    if not processing_service.database_enabled:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured")


def _get_recording_or_404(recording_id: str) -> dict[str, object]:
    recording = processing_service.get_recording(recording_id)
    if recording is None:
        raise HTTPException(status_code=404, detail="measurement not found")
    return recording


def _get_project_or_404(project_id: int) -> dict[str, object]:
    project = recording_repository.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


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

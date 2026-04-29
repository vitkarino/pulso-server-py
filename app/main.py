from datetime import UTC, date, datetime, time
import csv
from io import StringIO
import json
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import settings
from app.measurements import RecordingMetadata
from app.processing.service import PPGProcessingService
from app.realtime.store import metrics_store
from app.realtime.websocket import WebSocketController
from app.schemas.measurements import (
    MeasurementStartRequest,
    MeasurementState,
    ProjectPatchRequest,
    UserPatchRequest,
)
from app.schemas.metrics import VitalSigns
from app.storage.recording_repository import RecordingRepository

app = FastAPI(
    title="Pulso PPG Backend",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
recording_repository = RecordingRepository(settings.database_url)
processing_service = PPGProcessingService(settings, metrics_store, recording_repository)


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
            _device_for_api(
                device_id=device_id,
                connection_status=status,
                metrics=latest_metrics.get(device_id),
            )
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
    return _api_success({"metrics": _metrics_for_api(metrics)})


@app.delete("/api/devices/{device_id}")
async def delete_api_device(device_id: str) -> JSONResponse:
    was_connected = device_id in set(ws_controller.connected_devices())
    had_metrics = metrics_store.get(device_id) is not None
    stopped = processing_service.stop_recording_for_device(device_id)

    stop_command_sent = False
    if was_connected:
        stop_command_sent = await ws_controller.send_stop(device_id=device_id)
    disconnected = await ws_controller.disconnect_device(device_id)
    metrics_deleted = metrics_store.delete(device_id)
    processing_service.forget_device(device_id)

    if not (was_connected or had_metrics or stopped is not None or disconnected or metrics_deleted):
        raise HTTPException(status_code=404, detail="device not found")

    return _api_success(
        {
            "deleted": True,
            "device_id": device_id,
            "stopped_measurement_id": stopped.id if stopped is not None else None,
            "stop_command_sent": stop_command_sent,
            "disconnected": disconnected,
        }
    )


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


@app.patch("/api/projects/{project_id}")
def patch_api_project(project_id: int, body: ProjectPatchRequest) -> JSONResponse:
    _require_database()
    values = _patch_values(body, non_nullable_fields={"title"})
    project = recording_repository.update_project(project_id, values)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return _api_success({"project": project})


@app.delete("/api/projects/{project_id}")
def delete_api_project(project_id: int) -> JSONResponse:
    _require_database()
    if not recording_repository.delete_project(project_id):
        raise HTTPException(status_code=404, detail="project not found")
    return _api_success({"deleted": True, "project_id": project_id})


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


@app.patch("/api/users/{user_id}")
def patch_api_user(user_id: int, body: UserPatchRequest) -> JSONResponse:
    _require_database()
    values = _patch_values(body, non_nullable_fields={"name"})
    user = recording_repository.update_user(user_id, values)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return _api_success({"user": user})


@app.delete("/api/users/{user_id}")
def delete_api_user(user_id: int) -> JSONResponse:
    _require_database()
    if not recording_repository.delete_user(user_id):
        raise HTTPException(status_code=404, detail="user not found")
    return _api_success({"deleted": True, "user_id": user_id})


@app.post("/api/devices/{device_id}/measurements")
async def start_api_measurement(
    device_id: str,
    body: MeasurementStartRequest = Body(default_factory=MeasurementStartRequest),
) -> JSONResponse:
    duration_seconds = body.duration_s or settings.measurement_duration_seconds
    user_name = body.user_name
    if user_name is None and device_id.startswith("sim-"):
        user_name = "tester"
    try:
        measurement = processing_service.start_recording(
            duration_seconds=duration_seconds,
            metadata=RecordingMetadata(
                user_name=user_name,
                user_id=body.user_id,
                project_name=body.project_name,
                project_id=body.project_id,
            ),
            device_id=device_id,
        )
    except ValueError as exc:
        status_code = 409 if str(exc) == "measurement is already running for device" else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    command_sent = await ws_controller.send_start(
        device_id=device_id,
        measurement_id=measurement.id or "",
        duration_seconds=duration_seconds,
    )
    if not command_sent:
        processing_service.stop_recording(measurement.id or "")
        raise HTTPException(status_code=409, detail="device did not acknowledge start command")

    return _api_success({"measurement": _measurement_for_api(measurement)})


@app.post("/api/measurements/{measurement_id}/stop")
async def stop_api_measurement(measurement_id: str) -> JSONResponse:
    stopped = processing_service.stop_recording(measurement_id)
    if stopped is not None:
        if stopped.device_id is not None:
            await ws_controller.send_stop(device_id=stopped.device_id)
        return _api_success({"measurement": _measurement_for_api(stopped)})

    inactive_state = processing_service.get_recording_state(measurement_id)
    if inactive_state is not None:
        raise HTTPException(status_code=409, detail="measurement is not active")

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
    raise HTTPException(status_code=409, detail="measurement is not active")


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


@app.delete("/api/measurements/{measurement_id}")
async def delete_api_measurement(measurement_id: str) -> JSONResponse:
    _require_database()
    measurement = _get_recording_or_404(measurement_id)
    deleted_session = processing_service.delete_recording(measurement_id)

    stop_command_sent = False
    device_id = _string_or_none(measurement.get("device_id"))
    was_running = measurement.get("status") == "running"
    if deleted_session is not None:
        device_id = deleted_session.device_id or device_id
        was_running = was_running or deleted_session.status == "running"
    if was_running and device_id is not None:
        stop_command_sent = await ws_controller.send_stop(device_id=device_id)

    return _api_success(
        {
            "deleted": True,
            "measurement_id": measurement_id,
            "stop_command_sent": stop_command_sent,
        }
    )


@app.websocket("/ws/devices/{device_id}")
async def device_websocket_endpoint(websocket: WebSocket, device_id: str) -> None:
    await ws_controller.handle_device(websocket, device_id)


@app.websocket("/ws/measurements/{measurement_id}/stream")
async def measurement_stream_websocket_endpoint(websocket: WebSocket, measurement_id: str) -> None:
    await ws_controller.handle_measurement_stream(websocket, measurement_id)


JSON_ENCODERS = {datetime: lambda value: _datetime_for_api(value)}


def _api_success(data: object, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder(
            {
                "data": data,
                "$meta": _meta("success"),
            },
            custom_encoder=JSON_ENCODERS,
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
            },
            custom_encoder=JSON_ENCODERS,
        ),
        status_code=http_status,
    )


def _meta(status: Literal["success", "error"], *, error: dict[str, object] | None = None) -> dict[str, object]:
    now = datetime.now(UTC)
    meta: dict[str, object] = {
        "status": status,
        "time": {
            "iso": _datetime_for_api(now),
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
    if isinstance(measurement, MeasurementState):
        result = measurement.result
        finished_at = measurement.completed_at
        duration_ms = None
        if finished_at is not None:
            duration_ms = int(round((finished_at - measurement.started_at).total_seconds() * 1000))
        device_id = measurement.device_id
        if device_id is None and result is not None:
            device_id = result.device_id
        return {
            "id": measurement.id,
            "is_simulated": _is_simulated(device_id),
            "user_id": measurement.user_id,
            "project_id": measurement.project_id,
            "device_id": device_id,
            "time": {
                "started_at": _datetime_for_api(measurement.started_at),
                "finished_at": _datetime_for_api(finished_at),
                "duration_ms": duration_ms,
            },
            "status": INTERNAL_TO_PUBLIC_STATUS.get(measurement.status, measurement.status),
            "channels": ["ir", "red"],
            "sample_rate": result.fs if result is not None else None,
            "sensor_temp": result.temperature if result is not None else None,
            "bpm": result.bpm if result is not None else None,
            "spo2": result.spo2 if result is not None else None,
            "ratio": result.ratio if result is not None else None,
            "sensor_confidence": result.sensor_confidence if result is not None else None,
            "signal_quality": _signal_quality_for_api(result.signal_quality if result is not None else None),
        }

    payload = dict(measurement) if isinstance(measurement, dict) else jsonable_encoder(
        measurement,
        custom_encoder=JSON_ENCODERS,
    )
    if not isinstance(payload, dict):
        return payload

    device_id = _string_or_none(payload.get("device_id"))
    return {
        "id": payload.get("id"),
        "is_simulated": _is_simulated(device_id),
        "user_id": payload.get("user_id"),
        "project_id": payload.get("project_id"),
        "device_id": device_id,
        "time": {
            "started_at": _datetime_for_api(payload.get("started_at")),
            "finished_at": _datetime_for_api(payload.get("finished_at")),
            "duration_ms": payload.get("duration_ms"),
        },
        "status": _public_status(payload.get("status")),
        "channels": _channels_for_api(payload.get("signal_type")),
        "sample_rate": payload.get("sample_rate"),
        "sensor_temp": payload.get("sensor_temp"),
        "bpm": payload.get("bpm"),
        "spo2": payload.get("spo2"),
        "ratio": payload.get("ratio"),
        "sensor_confidence": payload.get("sensor_confidence"),
        "signal_quality": _signal_quality_for_api(payload.get("signal_quality")),
    }


def _device_for_api(
    *,
    device_id: str,
    connection_status: Literal["connected", "disconnected"],
    metrics: VitalSigns | None,
) -> dict[str, object]:
    return {
        "id": device_id,
        "is_simulated": _is_simulated(device_id),
        "connection_status": connection_status,
        "metrics": _metrics_for_api(metrics) if metrics is not None else None,
    }


def _metrics_for_api(metrics: VitalSigns) -> dict[str, object]:
    return {
        "device_id": metrics.device_id,
        "time": {"measured_at": _datetime_for_api(metrics.timestamp)},
        "sample_rate": metrics.fs,
        "sensor_temp": metrics.temperature,
        "bpm": metrics.bpm,
        "spo2": metrics.spo2,
        "ratio": metrics.ratio,
        "sensor_confidence": metrics.sensor_confidence,
        "signal_quality": _signal_quality_for_api(metrics.signal_quality),
    }


def _signal_quality_for_api(signal_quality: object | None) -> dict[str, object] | None:
    if signal_quality is None:
        return None
    payload = jsonable_encoder(signal_quality)
    if not isinstance(payload, dict):
        return None
    return {
        "level": payload.get("level"),
        "reason": payload.get("reason"),
        "peak_count": payload.get("peak_count"),
        "window_seconds": payload.get("window_seconds"),
        "perfusion_index": payload.get("perfusion_index"),
        "samples_in_window": payload.get("samples_in_window"),
    }


def _datetime_for_api(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        return _datetime_for_api(parsed)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return str(value)


def _public_status(status: object) -> object:
    if isinstance(status, str):
        return INTERNAL_TO_PUBLIC_STATUS.get(status, status)
    return status


def _channels_for_api(signal_type: object) -> list[str]:
    if signal_type == "IR":
        return ["ir"]
    if signal_type == "R":
        return ["red"]
    return ["ir", "red"]


def _is_simulated(device_id: str | None) -> bool:
    return bool(device_id and device_id.startswith("sim-"))


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _patch_values(body: object, *, non_nullable_fields: set[str]) -> dict[str, object]:
    values = body.model_dump(exclude_unset=True) if hasattr(body, "model_dump") else {}
    if not isinstance(values, dict):
        raise HTTPException(status_code=400, detail="request body must be an object")

    for field_name in non_nullable_fields:
        if field_name in values and values[field_name] is None:
            raise HTTPException(status_code=422, detail=f"{field_name} must not be null")
    return values


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
        return _datetime_for_api(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


ws_controller = WebSocketController(processing_service, metrics_formatter=_metrics_for_api)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.ws_port)

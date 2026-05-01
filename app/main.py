from __future__ import annotations

from datetime import UTC, date, datetime, time
import csv
from io import StringIO
import json
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.config import settings
from app.core.ids import (
    internal_device_id,
    is_simulated_device,
    public_device_id,
    public_measurement_id,
    public_project_id,
    public_quality_id,
    public_recording_id,
    public_user_id,
)
from app.measurements import RecordingMetadata
from app.processing.quality import QualityAnalysisResult, QualityModelUnavailable
from app.processing.service import PPGProcessingService
from app.realtime.store import metrics_store
from app.realtime.websocket import WebSocketController
from app.schemas.measurements import (
    MeasurementStartRequest,
    MeasurementState,
    ProjectCreateRequest,
    ProjectPatchRequest,
    RecordingStartRequest,
    UserCreateRequest,
    UserPatchRequest,
)
from app.schemas.metrics import VitalSigns
from app.storage.recording_repository import RecordingRepository

app = FastAPI(
    title="Pulso PPG Backend",
    version="0.4.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
recording_repository = RecordingRepository(settings.database_url)
processing_service = PPGProcessingService(settings, metrics_store, recording_repository)


RECORDING_EXPORT_FIELDS = [
    "id",
    "measurement_id",
    "quality_analysis_id",
    "use_for_ml_training",
    "status",
    "started_at",
    "finished_at",
    "duration_ms",
    "sample_start_index",
    "sample_end_index",
    "samples_count",
    "user_id",
    "project_id",
    "device_id",
    "sample_rate",
    "sensor_temp",
    "bpm",
    "spo2",
    "ratio",
    "peak_count",
]
SAMPLE_EXPORT_FIELDS = ["sample_index", "raw_ir", "raw_red", "raw_data"]


@app.on_event("startup")
def startup() -> None:
    processing_service.prepare_database()


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    code, message = _error_payload(exc.detail)
    return _api_error(exc.status_code, code, message)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    return _api_error(422, "validation_error", exc.errors()[0]["msg"])


@app.get("/api/health")
def api_health() -> JSONResponse:
    database_status = "connected" if processing_service.database_enabled else "disconnected"
    status = "ok" if database_status == "connected" else "degraded"
    return _api_success(
        {
            "status": status,
            "timestamp": _datetime_for_api(datetime.now(UTC)),
            "services": {
                "api": "ok",
                "database": database_status,
                "websocket": "ok",
            },
        }
    )


@app.get("/api/devices")
def list_devices(
    connection_status: Literal["connected", "disconnected"] | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    connected = {internal_device_id(device_id) or device_id for device_id in ws_controller.connected_devices()}
    latest_metrics = metrics_store.all()
    active_measurements = processing_service.get_measurements()
    known_device_ids = connected | set(latest_metrics) | set(active_measurements)

    devices = []
    for device_id in sorted(device_id for device_id in known_device_ids if device_id is not None):
        status = "connected" if device_id in connected else "disconnected"
        if connection_status is not None and status != connection_status:
            continue
        devices.append(
            _device_for_api(
                device_id=device_id,
                connection_status=status,
                measurement=active_measurements.get(device_id),
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
    raw_device_id = internal_device_id(device_id) or device_id
    metrics = metrics_store.get(raw_device_id)
    if metrics is None:
        _raise(404, "device_not_found", "Device metrics not found")
    return _api_success({"metrics": _metrics_for_api(metrics)})


@app.delete("/api/devices/{device_id}")
async def delete_api_device(device_id: str) -> JSONResponse:
    raw_device_id = internal_device_id(device_id) or device_id
    was_connected = raw_device_id in {internal_device_id(item) or item for item in ws_controller.connected_devices()}
    had_metrics = metrics_store.get(raw_device_id) is not None
    stopped = processing_service.stop_recording_for_device(raw_device_id)

    stop_command_sent = False
    if was_connected:
        stop_command_sent = await ws_controller.send_stop(device_id=raw_device_id)
    disconnected = await ws_controller.disconnect_device(raw_device_id)
    metrics_deleted = metrics_store.delete(raw_device_id)
    processing_service.forget_device(raw_device_id)

    if not (was_connected or had_metrics or stopped is not None or disconnected or metrics_deleted):
        _raise(404, "device_not_found", "Device not found")

    return _api_success(
        {
            "deleted": True,
            "device_id": public_device_id(raw_device_id),
            "stopped_measurement_id": stopped.id if stopped is not None else None,
            "stop_command_sent": stop_command_sent,
            "disconnected": disconnected,
        }
    )


@app.post("/api/projects")
def create_api_project(body: ProjectCreateRequest) -> JSONResponse:
    _require_database()
    return _api_success({"project": _project_for_api(recording_repository.create_project(body.model_dump()))}, status_code=201)


@app.get("/api/projects")
def list_projects(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    _require_database()
    return _api_success(
        {
            "projects": [_project_for_api(project) for project in recording_repository.list_projects(limit=limit, offset=offset)],
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/projects/{project_id}")
def get_api_project(project_id: str) -> JSONResponse:
    _require_database()
    return _api_success({"project": _project_for_api(_get_project_or_404(project_id))})


@app.patch("/api/projects/{project_id}")
def patch_api_project(project_id: str, body: ProjectPatchRequest) -> JSONResponse:
    _require_database()
    values = _patch_values(body, non_nullable_fields={"title"})
    project = recording_repository.update_project(project_id, values)
    if project is None:
        _raise(404, "project_not_found", "Project not found")
    return _api_success({"project": _project_for_api(project)})


@app.delete("/api/projects/{project_id}")
def delete_api_project(project_id: str) -> JSONResponse:
    _require_database()
    if not recording_repository.delete_project(project_id):
        _raise(404, "project_not_found", "Project not found")
    return _api_success({"deleted": True, "project_id": public_project_id(project_id)})


@app.post("/api/users")
def create_api_user(body: UserCreateRequest) -> JSONResponse:
    _require_database()
    return _api_success({"user": _user_for_api(recording_repository.create_user(body.model_dump()))}, status_code=201)


@app.get("/api/users")
def list_users(
    project_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    _require_database()
    if project_id is not None:
        _get_project_or_404(project_id)
    return _api_success(
        {
            "users": [_user_for_api(user) for user in recording_repository.list_users(project_id=project_id, limit=limit, offset=offset)],
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/users/{user_id}")
def get_api_user(user_id: str) -> JSONResponse:
    _require_database()
    return _api_success({"user": _user_for_api(_get_user_or_404(user_id))})


@app.patch("/api/users/{user_id}")
def patch_api_user(user_id: str, body: UserPatchRequest) -> JSONResponse:
    _require_database()
    values = _patch_values(body, non_nullable_fields={"name"})
    user = recording_repository.update_user(user_id, values)
    if user is None:
        _raise(404, "user_not_found", "User not found")
    return _api_success({"user": _user_for_api(user)})


@app.delete("/api/users/{user_id}")
def delete_api_user(user_id: str) -> JSONResponse:
    _require_database()
    if not recording_repository.delete_user(user_id):
        _raise(404, "user_not_found", "User not found")
    return _api_success({"deleted": True, "user_id": public_user_id(user_id)})


@app.post("/api/devices/{device_id}/measurements")
async def start_api_measurement(
    device_id: str,
    body: MeasurementStartRequest = Body(default_factory=MeasurementStartRequest),
) -> JSONResponse:
    raw_device_id = internal_device_id(device_id) or device_id
    if body.user_id is not None:
        _get_user_or_404(body.user_id)
    if body.project_id is not None:
        _get_project_or_404(body.project_id)
    try:
        measurement = processing_service.start_live_measurement(
            duration_seconds=body.duration_s,
            metadata=RecordingMetadata(
                user_id=public_user_id(body.user_id),
                project_id=public_project_id(body.project_id),
            ),
            device_id=raw_device_id,
        )
    except ValueError as exc:
        if str(exc) == "measurement is already running for device":
            _raise(409, "measurement_already_running", "Measurement is already running")
        _raise(400, "validation_error", str(exc))

    command_sent = await ws_controller.send_start(
        device_id=raw_device_id,
        measurement_id=measurement.id or "",
        duration_seconds=body.duration_s,
    )
    if not command_sent:
        processing_service.stop_measurement(measurement.id or "")
        _raise(409, "device_disconnected", "Device did not acknowledge start command")

    return _api_success({"measurement": _measurement_for_api(measurement)}, status_code=201)


@app.post("/api/measurements/{measurement_id}/stop")
async def stop_api_measurement(measurement_id: str) -> JSONResponse:
    stopped = processing_service.stop_measurement(measurement_id)
    if stopped is None:
        persisted = recording_repository.get_measurement(measurement_id) if processing_service.database_enabled else None
        if persisted is None:
            _raise(404, "measurement_not_found", "Measurement not found")
        _raise(409, "invalid_status_transition", "Measurement is not active")

    measurement, event = stopped
    raw_device_id = internal_device_id(measurement.device_id)
    if raw_device_id is not None:
        await ws_controller.send_stop(device_id=raw_device_id)
    if event is not None:
        await ws_controller.broadcast_recording_event(event)
    await ws_controller.broadcast_measurement_finished(measurement)
    return _api_success({"measurement": _measurement_for_api(measurement)})


@app.post("/api/measurements/{measurement_id}/recording")
async def start_api_recording(
    measurement_id: str,
    body: RecordingStartRequest = Body(default_factory=RecordingStartRequest),
) -> JSONResponse:
    _require_database()
    try:
        _measurement, event = processing_service.start_recording_for_measurement(
            measurement_id,
            duration_seconds=body.duration_s,
        )
    except KeyError:
        _raise(404, "measurement_not_found", "Measurement not found")
    except ValueError as exc:
        code = "recording_already_running" if str(exc) == "recording is already running" else "invalid_status_transition"
        _raise(409, code, str(exc))

    recording = _get_recording_or_404(event.recording_id)
    await ws_controller.broadcast_recording_event(event)
    return _api_success({"recording": _recording_for_api(recording)}, status_code=201)


@app.post("/api/measurements/{measurement_id}/recording/stop")
async def stop_api_recording(measurement_id: str) -> JSONResponse:
    _require_database()
    stopped = processing_service.stop_recording_for_measurement(measurement_id)
    if stopped is None:
        if recording_repository.get_measurement(measurement_id) is None:
            _raise(404, "measurement_not_found", "Measurement not found")
        _raise(409, "invalid_status_transition", "Recording is not active")
    _measurement, event = stopped
    recording = _get_recording_or_404(event.recording_id)
    await ws_controller.broadcast_recording_event(event)
    return _api_success({"recording": _recording_for_api(recording)})


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
    measurements = processing_service.list_measurements(
        limit=limit,
        offset=offset,
        date_from=_parse_date_bound(date_from, end_of_day=False),
        date_to=_parse_date_bound(date_to, end_of_day=True),
        device_id=device_id,
        user_id=user_id,
        project_id=project_id,
        status=status,
    )
    return _api_success(
        {
            "measurements": [_measurement_for_api(measurement) for measurement in measurements],
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/measurements/{measurement_id}")
def get_api_measurement(
    measurement_id: str,
    device_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    status: Literal["running", "completed", "failed", "cancelled"] | None = Query(default=None),
) -> JSONResponse:
    _require_database()
    measurement = _get_measurement_or_404(measurement_id)
    if not _measurement_matches_filters(
        measurement,
        date_from=_parse_date_bound(date_from, end_of_day=False),
        date_to=_parse_date_bound(date_to, end_of_day=True),
        device_id=device_id,
        user_id=user_id,
        project_id=project_id,
        status=status,
    ):
        _raise(404, "measurement_not_found", "Measurement not found")
    return _api_success({"measurement": _measurement_for_api(measurement)})


@app.get("/api/recordings")
def list_api_recordings(
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
    recordings = processing_service.list_recordings(
        limit=limit,
        offset=offset,
        date_from=_parse_date_bound(date_from, end_of_day=False),
        date_to=_parse_date_bound(date_to, end_of_day=True),
        device_id=device_id,
        user_id=user_id,
        project_id=project_id,
        status=status,
    )
    return _api_success(
        {
            "recordings": [_recording_for_api(recording) for recording in recordings],
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/recordings/{recording_id}/export")
def export_api_recording(
    recording_id: str,
    export_format: Literal["json", "csv"] = Query(default="json", alias="format"),
) -> object:
    _require_database()
    recording = _get_recording_or_404(recording_id)
    samples = processing_service.get_recording_samples(recording_id, limit=None, offset=0)

    if export_format == "csv":
        return _recordings_csv_response(
            f"{_public_recording_id_from_row(recording)}.csv",
            [recording],
            {str(recording["id"]): samples},
        )

    return _api_success(
        {
            "recording": _recording_for_api(recording),
            "samples": samples,
        }
    )


@app.get("/api/recordings/{recording_id}/samples")
def get_api_recording_samples(
    recording_id: str,
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    _require_database()
    recording = _get_recording_or_404(recording_id)
    return _api_success(
        {
            "recording_id": _public_recording_id_from_row(recording),
            "samples": processing_service.get_recording_samples_for_api(
                recording_id,
                limit=limit,
                offset=offset,
            ),
            "limit": limit,
            "offset": offset,
        }
    )


@app.get("/api/recordings/{recording_id}")
def get_api_recording(recording_id: str) -> JSONResponse:
    _require_database()
    return _api_success({"recording": _recording_for_api(_get_recording_or_404(recording_id))})


@app.delete("/api/recordings/{recording_id}")
async def delete_api_recording(recording_id: str) -> JSONResponse:
    _require_database()
    recording = _get_recording_or_404(recording_id)
    deleted_session = processing_service.delete_recording(recording_id)

    stop_command_sent = False
    raw_device_id = internal_device_id(recording.get("device_id"))
    was_running = recording.get("status") == "running"
    if deleted_session is not None:
        raw_device_id = internal_device_id(deleted_session.device_id) or raw_device_id
        was_running = was_running or deleted_session.status == "running"
    if was_running and raw_device_id is not None:
        stop_command_sent = await ws_controller.send_stop(device_id=raw_device_id)

    return _api_success(
        {
            "deleted": True,
            "recording_id": _public_recording_id_from_row(recording),
            "stop_command_sent": stop_command_sent,
        }
    )


@app.post("/api/recordings/{recording_id}/quality-analysis")
def run_quality_analysis(recording_id: str) -> JSONResponse:
    _require_database()
    _get_recording_or_404(recording_id)
    try:
        result = processing_service.analyze_recording_quality(recording_id)
    except QualityModelUnavailable as exc:
        _raise(503, "quality_model_unavailable", str(exc))
    except ValueError as exc:
        _raise(409, "invalid_status_transition", str(exc))
    return _api_success(
        {
            "quality_analysis": _quality_analysis_for_api(
                result,
                recording_id=recording_id,
            )
        },
        status_code=201,
    )


@app.get("/api/recordings/{recording_id}/quality-analysis")
def get_quality_analysis(recording_id: str) -> JSONResponse:
    _require_database()
    recording = _get_recording_or_404(recording_id)
    analysis = processing_service.get_quality_analysis(recording_id)
    if analysis is None:
        _raise(404, "quality_analysis_not_found", "Quality analysis not found")
    return _api_success(
        {
            "quality_analysis": _quality_analysis_for_api(
                analysis,
                recording_id=_public_recording_id_from_row(recording),
            )
        }
    )


@app.websocket("/ws/devices/{device_id}")
async def device_websocket_endpoint(websocket: WebSocket, device_id: str) -> None:
    await ws_controller.handle_device(websocket, internal_device_id(device_id) or device_id)


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
    meta: dict[str, object] = {
        "status": status,
        "timestamp": _datetime_for_api(datetime.now(UTC)),
    }
    if error is not None:
        meta["error"] = error
    return meta


def _raise(status_code: int, code: str, message: str) -> None:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


def _error_payload(detail: object) -> tuple[str, str]:
    if isinstance(detail, dict):
        code = detail.get("code")
        message = detail.get("message")
        if isinstance(code, str) and isinstance(message, str):
            return code, message
    if isinstance(detail, str):
        return _fallback_error_code(detail), detail
    return "request_error", str(detail)


def _fallback_error_code(detail: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in detail)
    return "_".join(part for part in normalized.split("_") if part) or "request_error"


def _measurement_for_api(measurement: object) -> dict[str, object]:
    if isinstance(measurement, MeasurementState):
        result = measurement.result
        finished_at = measurement.completed_at
        duration_ms = None
        if finished_at is not None:
            duration_ms = int(round((finished_at - measurement.started_at).total_seconds() * 1000))
        device_id = internal_device_id(measurement.device_id)
        if device_id is None and result is not None:
            device_id = result.device_id
        return {
            "id": public_measurement_id(measurement.id),
            "is_simulated": is_simulated_device(device_id),
            "user_id": public_user_id(measurement.user_id),
            "project_id": public_project_id(measurement.project_id),
            "active_recording_id": public_recording_id(measurement.active_recording_id),
            "device_id": public_device_id(device_id),
            "time": {
                "started_at": _datetime_for_api(measurement.started_at),
                "finished_at": _datetime_for_api(finished_at),
                "duration_ms": duration_ms,
            },
            "status": measurement.status,
            "channels": ["ir", "red"],
            "sample_rate_hz": result.fs if result is not None else None,
            "sensor_temp_c": result.temperature if result is not None else None,
            "bpm": result.bpm if result is not None else None,
            "spo2": result.spo2 if result is not None else None,
            "ratio": result.ratio if result is not None else None,
        }

    payload = dict(measurement) if isinstance(measurement, dict) else jsonable_encoder(
        measurement,
        custom_encoder=JSON_ENCODERS,
    )
    device_id = internal_device_id(payload.get("device_id"))
    return {
        "id": payload.get("public_id") or public_measurement_id(payload.get("id")),
        "is_simulated": is_simulated_device(device_id),
        "user_id": public_user_id(payload.get("user_id")),
        "project_id": public_project_id(payload.get("project_id")),
        "active_recording_id": public_recording_id(payload.get("active_recording_id")),
        "device_id": public_device_id(device_id),
        "time": {
            "started_at": _datetime_for_api(payload.get("started_at")),
            "finished_at": _datetime_for_api(payload.get("finished_at")),
            "duration_ms": payload.get("duration_ms"),
        },
        "status": payload.get("status"),
        "channels": _channels_for_api(payload.get("signal_type")),
        "sample_rate_hz": payload.get("sample_rate"),
        "sensor_temp_c": payload.get("sensor_temp"),
        "bpm": payload.get("bpm"),
        "spo2": payload.get("spo2"),
        "ratio": payload.get("ratio"),
    }


def _recording_for_api(recording: dict[str, Any]) -> dict[str, object]:
    return {
        "id": _public_recording_id_from_row(recording),
        "measurement_id": public_measurement_id(recording.get("measurement_id")),
        "quality_analysis_id": public_quality_id(recording.get("quality_analysis_id")),
        "use_for_ml_training": bool(recording.get("use_for_ml_training")),
        "status": recording.get("status"),
        "time": {
            "started_at": _datetime_for_api(recording.get("started_at")),
            "finished_at": _datetime_for_api(recording.get("finished_at")),
            "duration_ms": recording.get("duration_ms"),
        },
        "sample_range": {
            "start_index": recording.get("sample_start_index"),
            "end_index": recording.get("sample_end_index"),
        },
        "samples_count": recording.get("samples_count"),
        "device_id": public_device_id(recording.get("device_id")),
        "user_id": public_user_id(recording.get("user_id")),
        "project_id": public_project_id(recording.get("project_id")),
        "sample_rate_hz": recording.get("sample_rate"),
        "sensor_temp_c": recording.get("sensor_temp"),
        "bpm": recording.get("bpm"),
        "spo2": recording.get("spo2"),
        "ratio": recording.get("ratio"),
    }


def _device_for_api(
    *,
    device_id: str,
    connection_status: Literal["connected", "disconnected"],
    measurement: MeasurementState | None,
) -> dict[str, object]:
    active_recording_id = measurement.active_recording_id if measurement is not None else None
    status = "idle"
    if measurement is not None and measurement.status == "running":
        status = "recording" if active_recording_id is not None else "measuring"
    return {
        "id": public_device_id(device_id),
        "is_simulated": is_simulated_device(device_id),
        "connection_status": connection_status,
        "status": status,
        "active_measurement_id": measurement.id if measurement is not None and measurement.status == "running" else None,
        "active_recording_id": active_recording_id,
    }


def _metrics_for_api(metrics: VitalSigns) -> dict[str, object]:
    return {
        "device_id": public_device_id(metrics.device_id),
        "timestamp": _datetime_for_api(metrics.timestamp),
        "sample_rate_hz": metrics.fs,
        "sensor_temp_c": metrics.temperature,
        "bpm": metrics.bpm,
        "spo2": metrics.spo2,
        "ratio": metrics.ratio,
        "live_quality": _live_quality_for_api(metrics),
    }


def _live_quality_for_api(metrics: VitalSigns) -> dict[str, object]:
    quality = metrics.signal_quality
    level = quality.level if quality.level in {"high", "medium"} else "low"
    return {
        "level": level,
        "is_recording_ready": level in {"high", "medium"},
        "reason": _quality_reason_for_api(quality.reason, level),
    }


def _quality_reason_for_api(reason: str | None, level: str) -> str | None:
    if level == "high":
        return None
    if reason is None:
        return "high_noise"
    lowered = reason.lower()
    if "saturat" in lowered or "exposed" in lowered:
        return "saturation"
    if "flat" in lowered or "no usable" in lowered:
        return "flatline"
    if "perfusion" in lowered or "dc level" in lowered:
        return "low_perfusion"
    if "peak" in lowered:
        return "irregular_rhythm"
    return "high_noise"


def _project_for_api(project: dict[str, Any]) -> dict[str, object]:
    return {
        "id": project.get("public_id") or public_project_id(project.get("id")),
        "title": project.get("title"),
        "description": project.get("description"),
        "users_count": project.get("users_count", 0),
        "recordings_count": project.get("recordings_qty", 0),
        "created_at": _datetime_for_api(project.get("created_at")),
    }


def _user_for_api(user: dict[str, Any]) -> dict[str, object]:
    return {
        "id": user.get("public_id") or public_user_id(user.get("id")),
        "name": user.get("name"),
        "age": user.get("age"),
        "sex": user.get("sex"),
        "projects_count": user.get("projects_count", 0),
        "recordings_count": user.get("recordings_qty", 0),
        "created_at": _datetime_for_api(user.get("created_at")),
    }


def _quality_analysis_for_api(
    analysis: QualityAnalysisResult | dict[str, Any],
    *,
    recording_id: str,
) -> dict[str, object]:
    if isinstance(analysis, QualityAnalysisResult):
        return {
            "id": analysis.public_id,
            "recording_id": public_recording_id(recording_id),
            "timestamp": _datetime_for_api(analysis.timestamp),
            "model": analysis.model,
            "quality_result": analysis.quality_result,
            "features": analysis.features,
        }
    return {
        "id": analysis.get("public_id") or public_quality_id(analysis.get("id")),
        "recording_id": public_recording_id(recording_id),
        "timestamp": _datetime_for_api(analysis.get("timestamp")),
        "model": analysis.get("model"),
        "quality_result": analysis.get("quality_result"),
        "features": analysis.get("features"),
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


def _channels_for_api(signal_type: object) -> list[str]:
    if signal_type == "IR":
        return ["ir"]
    if signal_type == "R":
        return ["red"]
    return ["ir", "red"]


def _patch_values(body: object, *, non_nullable_fields: set[str]) -> dict[str, object]:
    values = body.model_dump(exclude_unset=True) if hasattr(body, "model_dump") else {}
    if not isinstance(values, dict):
        _raise(400, "validation_error", "Request body must be an object")

    for field_name in non_nullable_fields:
        if field_name in values and values[field_name] is None:
            _raise(422, "validation_error", f"{field_name} must not be null")
    return values


def _require_database() -> None:
    if not processing_service.database_enabled:
        _raise(503, "database_unavailable", "DATABASE_URL is not configured")


def _get_recording_or_404(recording_id: str) -> dict[str, Any]:
    recording = processing_service.get_recording(recording_id)
    if recording is None:
        _raise(404, "recording_not_found", "Recording not found")
    return recording


def _get_measurement_or_404(measurement_id: str) -> dict[str, Any]:
    measurement = processing_service.get_measurement_record(measurement_id)
    if measurement is None:
        _raise(404, "measurement_not_found", "Measurement not found")
    return measurement


def _measurement_matches_filters(
    measurement: dict[str, Any],
    *,
    date_from: datetime | None,
    date_to: datetime | None,
    device_id: str | None,
    user_id: str | None,
    project_id: str | None,
    status: str | None,
) -> bool:
    started_at = measurement.get("started_at")
    if isinstance(started_at, str):
        try:
            started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        except ValueError:
            started_at = None
    if isinstance(started_at, datetime) and started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)

    if date_from is not None and isinstance(started_at, datetime) and started_at < date_from:
        return False
    if date_to is not None and isinstance(started_at, datetime) and started_at > date_to:
        return False
    if device_id is not None and measurement.get("device_id") != internal_device_id(device_id):
        return False
    if user_id is not None and measurement.get("user_id") != public_user_id(user_id):
        return False
    if project_id is not None and measurement.get("project_id") != public_project_id(project_id):
        return False
    if status is not None and measurement.get("status") != status:
        return False
    return True


def _get_project_or_404(project_id: str) -> dict[str, Any]:
    _require_database()
    project = recording_repository.get_project(project_id)
    if project is None:
        _raise(404, "project_not_found", "Project not found")
    return project


def _get_user_or_404(user_id: str) -> dict[str, Any]:
    _require_database()
    user = recording_repository.get_user(user_id)
    if user is None:
        _raise(404, "user_not_found", "User not found")
    return user


def _public_recording_id_from_row(recording: dict[str, Any]) -> str | None:
    return recording.get("public_id") or public_recording_id(recording.get("id"))


def _parse_date_bound(raw_value: str | None, *, end_of_day: bool) -> datetime | None:
    if raw_value is None or not isinstance(raw_value, str):
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
            detail={"code": "validation_error", "message": "date filters must use YYYY-MM-DD or ISO datetime format"},
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
    api_recording = _recording_for_api(dict(recording))
    row = {
        "id": api_recording.get("id"),
        "measurement_id": api_recording.get("measurement_id"),
        "quality_analysis_id": api_recording.get("quality_analysis_id"),
        "use_for_ml_training": api_recording.get("use_for_ml_training"),
        "status": api_recording.get("status"),
        "started_at": api_recording.get("time", {}).get("started_at") if isinstance(api_recording.get("time"), dict) else None,
        "finished_at": api_recording.get("time", {}).get("finished_at") if isinstance(api_recording.get("time"), dict) else None,
        "duration_ms": api_recording.get("time", {}).get("duration_ms") if isinstance(api_recording.get("time"), dict) else None,
        "sample_start_index": api_recording.get("sample_range", {}).get("start_index") if isinstance(api_recording.get("sample_range"), dict) else None,
        "sample_end_index": api_recording.get("sample_range", {}).get("end_index") if isinstance(api_recording.get("sample_range"), dict) else None,
        "samples_count": api_recording.get("samples_count"),
        "user_id": api_recording.get("user_id"),
        "project_id": api_recording.get("project_id"),
        "device_id": api_recording.get("device_id"),
        "sample_rate": api_recording.get("sample_rate_hz"),
        "sensor_temp": api_recording.get("sensor_temp_c"),
        "bpm": api_recording.get("bpm"),
        "spo2": api_recording.get("spo2"),
        "ratio": api_recording.get("ratio"),
        "peak_count": recording.get("peak_count"),
    }
    raw_data = sample.get("raw_data") if sample is not None else None
    raw = raw_data if isinstance(raw_data, dict) else {}
    row.update(
        {
            "sample_index": sample.get("sample_index") if sample is not None else None,
            "raw_ir": raw.get("ir"),
            "raw_red": raw.get("red", raw.get("r")),
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

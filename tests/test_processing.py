import asyncio
import json
import math
import pickle
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, inspect, text

from app import main as main_module
from app.core.config import AppConfig
from app.llm import AssistantService
from app.main import _api_success, _datetime_for_api, _measurement_for_api, _metrics_for_api
from app.measurements import RecordingMetadata
from app.processing.quality import extract_morphology, extract_quality_features
from app.processing.filters import PPGNoiseFilter
from app.processing.service import PPGProcessingService
from app.realtime.store import MetricsStore
from app.realtime.websocket import WebSocketController
from app.schemas.assistant import AssistantChatRequest
from app.schemas.measurements import (
    MeasurementStartRequest,
    ProjectCreateRequest,
    ProjectPatchRequest,
    ProjectUserAssignRequest,
    RecordingStartRequest,
    UserCreateRequest,
    UserPatchRequest,
)
from app.storage.recording_repository import RecordingRepository


class FakeQualityModel:
    classes_ = np.asarray(["low", "medium", "high"])

    def predict_proba(self, _values: np.ndarray) -> np.ndarray:
        return np.asarray([[0.02, 0.08, 0.90]])


class FakeLLMProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return "Запись качественная, поэтому BPM и SpO2 выглядят достаточно надежными."


class AckingController:
    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.stopped: list[str] = []
        self.recording_events: list[Any] = []
        self.finished: list[Any] = []

    def connected_devices(self) -> list[str]:
        return ["sim-api-device"]

    async def send_start(self, *, device_id: str, measurement_id: str, duration_seconds: float | None) -> bool:
        self.started.append(
            {
                "device_id": device_id,
                "measurement_id": measurement_id,
                "duration_seconds": duration_seconds,
            }
        )
        return True

    async def send_stop(self, *, device_id: str) -> bool:
        self.stopped.append(device_id)
        return True

    async def disconnect_device(self, _device_id: str) -> bool:
        return True

    async def broadcast_recording_event(self, event: Any) -> None:
        self.recording_events.append(event)

    async def broadcast_measurement_finished(self, measurement: Any) -> None:
        self.finished.append(measurement)


def _synthetic_payload(
    fs: float = 25,
    seconds: float = 18,
    bpm: float = 72.0,
    measurement_id: str | None = None,
    device_id: str = "dev_A0:B7:65:12:34:56",
    start_index: int = 0,
) -> str:
    count = int(math.ceil(fs * seconds - 1e-9))
    t = np.arange(count) / fs
    freq = bpm / 60.0
    ir_dc = 84_000.0
    red_dc = 53_000.0
    ir_ac = 1_500.0
    ratio = 0.55
    red_ac = ratio * ir_ac * red_dc / ir_dc
    rng = np.random.default_rng(42)

    ir = ir_dc + ir_ac * np.sin(2 * math.pi * freq * t) + rng.normal(0, 60, count)
    red = red_dc + red_ac * np.sin(2 * math.pi * freq * t) + rng.normal(0, 45, count)
    samples = [
        {"index": start_index + index, "ir": float(i), "red": float(r)}
        for index, (i, r) in enumerate(zip(ir, red, strict=True))
    ]

    return json.dumps(
        {
            "type": "samples",
            "device_id": device_id,
            "measurement_id": measurement_id,
            "sample_rate_hz": fs,
            "sensor_temp_c": 31.75,
            "samples": samples,
            "timestamp": "2026-04-28T11:51:20.778832Z",
        }
    )


async def _asgi_request(method: str, path: str, headers: dict[str, str]) -> dict[str, Any]:
    sent: list[dict[str, Any]] = []
    received = False
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [(name.lower().encode("ascii"), value.encode("ascii")) for name, value in headers.items()],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8080),
    }

    async def receive() -> dict[str, Any]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await main_module.app(scope, receive, send)
    response_start = next(message for message in sent if message["type"] == "http.response.start")
    response_headers = {
        name.decode("latin-1"): value.decode("latin-1")
        for name, value in response_start.get("headers", [])
    }
    return {"status": response_start["status"], "headers": response_headers}


def _raw_signal(fs: float = 25, seconds: float = 18, bpm: float = 72.0) -> tuple[np.ndarray, np.ndarray]:
    count = int(math.ceil(fs * seconds - 1e-9))
    t = np.arange(count) / fs
    pulse = np.sin(2 * math.pi * bpm / 60.0 * t)
    return 84_000.0 + 1_500.0 * pulse, 53_000.0 + 600.0 * pulse


class ProcessingTests(unittest.TestCase):
    def test_processing_accepts_v4_payload_and_estimates_metrics(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())

        metrics = service.process_json(_synthetic_payload(measurement_id="mes_demo"))

        self.assertEqual(metrics.device_id, "A0:B7:65:12:34:56")
        self.assertIsNotNone(metrics.bpm)
        self.assertIsNotNone(metrics.spo2)
        self.assertIn(metrics.signal_quality.level, {"medium", "high"})

    def test_start_measurement_keeps_default_fixed_duration_for_internal_api(self) -> None:
        service = PPGProcessingService(AppConfig(measurement_duration_seconds=15.0), MetricsStore())
        started = service.start_measurement("A0:B7:65:12:34:56")
        assert started.id is not None

        service.process_json(_synthetic_payload(seconds=18, measurement_id=started.id))
        measurement = service.get_measurement("A0:B7:65:12:34:56")

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertEqual(measurement.status, "completed")
        self.assertEqual(measurement.samples_collected, 375)
        self.assertIsNotNone(measurement.result)

    def test_api_formatters_use_v4_shape(self) -> None:
        service = PPGProcessingService(AppConfig(measurement_duration_seconds=15.0), MetricsStore())
        metrics = service.process_json(_synthetic_payload())
        metrics_payload = _metrics_for_api(metrics)

        self.assertEqual(metrics_payload["device_id"], "dev_A0:B7:65:12:34:56")
        self.assertEqual(metrics_payload["sample_rate_hz"], 25)
        self.assertEqual(metrics_payload["sensor_temp_c"], 31.75)
        self.assertIn("live_quality", metrics_payload)
        self.assertNotIn("signal_quality", metrics_payload)

        started = service.start_recording(
            duration_seconds=15.0,
            metadata=RecordingMetadata(user_id="1", project_id="1"),
            device_id="sim-device-001",
        )
        assert started.id is not None
        service.process_json(_synthetic_payload(seconds=18, measurement_id=started.id, device_id="dev_sim-device-001"))
        measurement = service.get_measurement("sim-device-001")
        assert measurement is not None
        measurement_payload = _measurement_for_api(measurement)

        self.assertTrue(str(measurement_payload["id"]).startswith("mes_"))
        self.assertEqual(measurement_payload["user_id"], "usr_1")
        self.assertEqual(measurement_payload["project_id"], "prj_1")
        self.assertEqual(measurement_payload["device_id"], "dev_sim-device-001")
        self.assertEqual(measurement_payload["sample_rate_hz"], 25)
        self.assertNotIn("sample_rate", measurement_payload)

    def test_api_success_uses_v4_meta_timestamp(self) -> None:
        response = _api_success({"created_at": datetime(2026, 4, 28, 11, 51, 20, tzinfo=UTC)})
        payload = json.loads(response.body)

        self.assertEqual(payload["data"]["created_at"], "2026-04-28T11:51:20Z")
        self.assertTrue(payload["$meta"]["timestamp"].endswith("Z"))
        self.assertNotIn("time", payload["$meta"])


class MorphologyAndFeatureTests(unittest.TestCase):
    def test_morphology_extracts_stable_template_for_clean_signal(self) -> None:
        config = AppConfig()
        raw_ir, raw_red = _raw_signal()
        filtered_ir, _filtered_red = PPGNoiseFilter(config).filter_pair(raw_ir, raw_red, 25)

        morphology = extract_morphology(filtered_ir=filtered_ir, fs=25, config=config)
        features = extract_quality_features(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            fs=25,
            config=config,
            morphology=morphology,
        )

        self.assertGreaterEqual(morphology.valid_pulse_count, 3)
        self.assertEqual(len(morphology.average_pulse_template), 100)
        self.assertGreater(features["template_corr_mean"], 0.8)
        self.assertGreater(features["relative_power_hr_band"], 0.5)

    def test_morphology_handles_flatline_and_saturation_features(self) -> None:
        config = AppConfig(max_sensor_dc=250_000.0)
        raw_ir = np.full(300, 260_000.0)
        raw_red = np.full(300, 260_000.0)
        filtered_ir = np.zeros(300)

        morphology = extract_morphology(filtered_ir=filtered_ir, fs=25, config=config)
        features = extract_quality_features(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            fs=25,
            config=config,
            morphology=morphology,
        )

        self.assertEqual(morphology.shape_quality, "low_amplitude")
        self.assertGreater(features["flatline_ratio"], 0.8)
        self.assertEqual(features["saturation_ratio"], 1.0)

    def test_features_report_outliers_for_noisy_spike(self) -> None:
        config = AppConfig()
        raw_ir, raw_red = _raw_signal()
        filtered_ir = raw_ir - np.mean(raw_ir)
        filtered_ir[100] += 30_000

        features = extract_quality_features(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            fs=25,
            config=config,
        )

        self.assertGreater(features["outlier_ratio"], 0.0)


class WebSocketControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_start_waits_for_start_ack(self) -> None:
        controller = WebSocketController(object(), ack_timeout_seconds=0.05)

        class AckingWebSocket:
            async def send_json(self, payload: dict[str, Any]) -> None:
                await controller._handle_device_control_message(
                    json.dumps(
                        {
                            "type": "start_ack",
                            "measurement_id": payload["measurement_id"],
                        }
                    ),
                    "sim-test",
                )

        controller._connections["sim-test"] = AckingWebSocket()

        sent = await controller.send_start(
            device_id="sim-test",
            measurement_id="mes_measurement-1",
            duration_seconds=10.0,
        )

        self.assertTrue(sent)

    async def test_live_sample_payload_uses_measurement_update_shape(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())
        started = service.start_measurement("A0:B7:65:12:34:56")
        assert started.id is not None
        raw_payload = _synthetic_payload(seconds=1, measurement_id=started.id)

        processed = service.process_json_with_recordings_and_signal(raw_payload)
        controller = WebSocketController(service, metrics_formatter=_metrics_for_api)
        live_payload = controller._live_sample_payload(
            raw_payload,
            processed.metrics,
            processed_samples=processed.processed_samples,
        )

        self.assertIsNotNone(live_payload)
        assert live_payload is not None
        self.assertEqual(live_payload["type"], "measurement_update")
        self.assertEqual(live_payload["measurement_id"], started.id)
        self.assertEqual(live_payload["device_id"], "dev_A0:B7:65:12:34:56")
        self.assertEqual(live_payload["sample_rate_hz"], 25)
        self.assertEqual(live_payload["sensor_temp_c"], 31.75)
        self.assertEqual(set(live_payload["samples"][0]), {"index", "t_ms", "ir", "red", "ir_filtered", "red_filtered"})
        self.assertIn("live_quality", live_payload["metrics"])


class ApiMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_repository = main_module.recording_repository
        self._old_processing_service = main_module.processing_service
        self._old_assistant_service = main_module.assistant_service
        self._old_ws_controller = main_module.ws_controller
        self._old_metrics_store = main_module.metrics_store
        self._tempdir = tempfile.TemporaryDirectory()
        self.database_url = f"sqlite:///{Path(self._tempdir.name) / 'test.db'}"
        self.repository = RecordingRepository(self.database_url)
        self.repository.create_schema()
        self.metrics_store = MetricsStore()
        self.processing_service = PPGProcessingService(
            AppConfig(database_url=self.database_url),
            self.metrics_store,
            self.repository,
        )
        self.ws_controller = AckingController()
        main_module.recording_repository = self.repository
        main_module.processing_service = self.processing_service
        main_module.assistant_service = AssistantService(AppConfig())
        main_module.metrics_store = self.metrics_store
        main_module.ws_controller = self.ws_controller

    def tearDown(self) -> None:
        main_module.recording_repository = self._old_repository
        main_module.processing_service = self._old_processing_service
        main_module.assistant_service = self._old_assistant_service
        main_module.ws_controller = self._old_ws_controller
        main_module.metrics_store = self._old_metrics_store
        self.repository.dispose()
        self._tempdir.cleanup()

    def _create_user_project(self) -> tuple[str, str]:
        user_response = main_module.create_api_user(UserCreateRequest(name="Alice", age=30, sex="female"))
        project_response = main_module.create_api_project(ProjectCreateRequest(title="Study", description="PPG"))
        user_id = json.loads(user_response.body)["data"]["user"]["id"]
        project_id = json.loads(project_response.body)["data"]["project"]["id"]
        return user_id, project_id

    def test_cors_preflight_allows_local_frontend(self) -> None:
        response = asyncio.run(
            _asgi_request(
                "OPTIONS",
                "/api/projects",
                {
                    "origin": "http://localhost:3000",
                    "access-control-request-method": "POST",
                    "access-control-request-headers": "content-type",
                },
            )
        )

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["headers"]["access-control-allow-origin"], "http://localhost:3000")
        self.assertIn("POST", response["headers"]["access-control-allow-methods"])

    def test_user_and_project_crud_use_public_ids(self) -> None:
        user_id, project_id = self._create_user_project()

        self.assertTrue(user_id.startswith("usr_"))
        self.assertTrue(project_id.startswith("prj_"))

        patched_user = json.loads(
            main_module.patch_api_user(user_id, UserPatchRequest(age=31)).body
        )["data"]["user"]
        patched_project = json.loads(
            main_module.patch_api_project(project_id, ProjectPatchRequest(description=None)).body
        )["data"]["project"]

        self.assertEqual(patched_user["age"], 31)
        self.assertIsNone(patched_project["description"])
        self.assertEqual(json.loads(main_module.get_api_user(user_id).body)["data"]["user"]["id"], user_id)
        self.assertEqual(json.loads(main_module.get_api_project(project_id).body)["data"]["project"]["id"], project_id)

    def test_project_user_assignment_routes(self) -> None:
        user_id, project_id = self._create_user_project()

        assign_response = main_module.assign_api_project_user(project_id, ProjectUserAssignRequest(user_id=user_id))
        assignment = json.loads(assign_response.body)["data"]["assignment"]

        self.assertEqual(assign_response.status_code, 201)
        self.assertEqual(assignment["project_id"], project_id)
        self.assertEqual(assignment["user_id"], user_id)
        self.assertIsNotNone(assignment["assigned_at"])

        listed_users = json.loads(main_module.list_users(project_id=project_id, limit=100, offset=0).body)["data"]["users"]
        project = json.loads(main_module.get_api_project(project_id).body)["data"]["project"]
        user = json.loads(main_module.get_api_user(user_id).body)["data"]["user"]

        self.assertEqual([listed_users[0]["id"]], [user_id])
        self.assertEqual(project["users_count"], 1)
        self.assertEqual(user["projects_count"], 1)

        with self.assertRaises(HTTPException) as context:
            main_module.assign_api_project_user(project_id, ProjectUserAssignRequest(user_id=user_id))

        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(context.exception.detail["code"], "project_user_already_exists")

        delete_response = main_module.delete_api_project_user(project_id, user_id)
        deleted = json.loads(delete_response.body)["data"]

        self.assertTrue(deleted["deleted"])
        self.assertEqual(deleted["project_id"], project_id)
        self.assertEqual(deleted["user_id"], user_id)
        self.assertEqual(json.loads(main_module.get_api_project(project_id).body)["data"]["project"]["users_count"], 0)

        with self.assertRaises(HTTPException) as context:
            main_module.delete_api_project_user(project_id, user_id)

        self.assertEqual(context.exception.status_code, 404)
        self.assertEqual(context.exception.detail["code"], "assignment_not_found")

    def test_measurement_recording_lifecycle_and_recording_routes(self) -> None:
        user_id, project_id = self._create_user_project()
        start_response = asyncio.run(
            main_module.start_api_measurement(
                "dev_sim-api-device",
                MeasurementStartRequest(user_id=user_id, project_id=project_id),
            )
        )
        measurement = json.loads(start_response.body)["data"]["measurement"]
        measurement_id = measurement["id"]

        self.assertTrue(measurement_id.startswith("mes_"))
        self.assertEqual(measurement["status"], "running")
        self.assertEqual(self.ws_controller.started[0]["measurement_id"], measurement_id)

        measurement_detail = json.loads(
            main_module.get_api_measurement(
                measurement_id,
                user_id=user_id,
                project_id=project_id,
                device_id="dev_sim-api-device",
                date_from=None,
                date_to=None,
                status=None,
            ).body
        )["data"]["measurement"]
        listed_measurements = json.loads(
            main_module.list_api_measurements(
                user_id=user_id,
                project_id=project_id,
                device_id="dev_sim-api-device",
                date_from=None,
                date_to=None,
                status="running",
                limit=100,
                offset=0,
            ).body
        )["data"]["measurements"]

        self.assertEqual(measurement_detail["id"], measurement_id)
        self.assertEqual(listed_measurements[0]["id"], measurement_id)

        recording_response = asyncio.run(
            main_module.start_api_recording(measurement_id, RecordingStartRequest(duration_s=None))
        )
        recording = json.loads(recording_response.body)["data"]["recording"]
        recording_id = recording["id"]
        self.assertTrue(recording_id.startswith("rec_"))
        self.assertEqual(recording["measurement_id"], measurement_id)

        self.processing_service.process_json(
            _synthetic_payload(seconds=18, measurement_id=measurement_id, device_id="dev_sim-api-device")
        )
        stop_response = asyncio.run(main_module.stop_api_recording(measurement_id))
        stopped_recording = json.loads(stop_response.body)["data"]["recording"]

        self.assertEqual(stopped_recording["status"], "completed")
        self.assertEqual(stopped_recording["sample_range"]["start_index"], 0)
        self.assertEqual(stopped_recording["samples_count"], 450)

        detail = json.loads(main_module.get_api_recording(recording_id).body)["data"]["recording"]
        samples = json.loads(main_module.get_api_recording_samples(recording_id, limit=5, offset=0).body)["data"]["samples"]
        listed = json.loads(
            main_module.list_api_recordings(
                user_id=user_id,
                project_id=project_id,
                device_id=None,
                date_from=None,
                date_to=None,
                status=None,
                limit=100,
                offset=0,
            ).body
        )["data"]["recordings"]

        self.assertEqual(detail["id"], recording_id)
        self.assertEqual(len(samples), 5)
        self.assertEqual(set(samples[0]), {"index", "t_ms", "ir", "red", "ir_filtered", "red_filtered"})
        self.assertEqual(listed[0]["id"], recording_id)

        export_response = main_module.export_api_recording(recording_id, export_format="json")
        export_payload = json.loads(export_response.body)
        self.assertEqual(export_payload["data"]["recording"]["id"], recording_id)

        delete_response = asyncio.run(main_module.delete_api_recording(recording_id))
        self.assertTrue(json.loads(delete_response.body)["data"]["deleted"])
        self.assertIsNone(self.repository.get_recording(recording_id))

    def test_stop_measurement_rejects_inactive_measurement(self) -> None:
        user_id, project_id = self._create_user_project()
        measurement = json.loads(
            asyncio.run(
                main_module.start_api_measurement(
                    "dev_sim-api-device",
                    MeasurementStartRequest(user_id=user_id, project_id=project_id),
                )
            ).body
        )["data"]["measurement"]

        first_response = asyncio.run(main_module.stop_api_measurement(measurement["id"]))
        self.assertEqual(json.loads(first_response.body)["data"]["measurement"]["status"], "cancelled")

        with self.assertRaises(HTTPException) as context:
            asyncio.run(main_module.stop_api_measurement(measurement["id"]))

        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(context.exception.detail["code"], "invalid_status_transition")

    def test_quality_analysis_requires_model_and_stores_model_result(self) -> None:
        user_id, project_id = self._create_user_project()
        measurement = json.loads(
            asyncio.run(
                main_module.start_api_measurement(
                    "dev_sim-api-device",
                    MeasurementStartRequest(user_id=user_id, project_id=project_id),
                )
            ).body
        )["data"]["measurement"]
        recording = json.loads(
            asyncio.run(main_module.start_api_recording(measurement["id"], RecordingStartRequest())).body
        )["data"]["recording"]
        self.processing_service.process_json(
            _synthetic_payload(seconds=18, measurement_id=measurement["id"], device_id="dev_sim-api-device")
        )
        asyncio.run(main_module.stop_api_recording(measurement["id"]))

        with self.assertRaises(HTTPException) as context:
            main_module.run_quality_analysis(recording["id"])
        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail["code"], "quality_model_unavailable")

        with self.assertRaises(HTTPException) as context:
            main_module.assistant_chat(AssistantChatRequest(recording_id=recording["id"], message="Что с записью?"))
        self.assertEqual(context.exception.status_code, 404)
        self.assertEqual(context.exception.detail["code"], "quality_analysis_not_found")

        model_path = Path(self._tempdir.name) / "quality.pkl"
        with model_path.open("wb") as model_file:
            pickle.dump(FakeQualityModel(), model_file)

        self.processing_service = PPGProcessingService(
            AppConfig(database_url=self.database_url, quality_model_path=str(model_path)),
            self.metrics_store,
            self.repository,
        )
        main_module.processing_service = self.processing_service

        response = main_module.run_quality_analysis(recording["id"])
        payload = json.loads(response.body)["data"]["quality_analysis"]

        self.assertTrue(payload["id"].startswith("qlt_"))
        self.assertEqual(payload["recording_id"], recording["id"])
        self.assertEqual(payload["quality_result"]["level"], "high")
        self.assertIn("template_corr_mean", payload["features"])

        stored = json.loads(main_module.get_quality_analysis(recording["id"]).body)["data"]["quality_analysis"]
        self.assertEqual(stored["id"], payload["id"])

        fake_provider = FakeLLMProvider()
        main_module.assistant_service = AssistantService(
            AppConfig(llm_enabled=True, llm_provider="ollama", llm_model="fake-local-model"),
            fake_provider,
        )
        assistant_response = main_module.assistant_chat(
            AssistantChatRequest(recording_id=recording["id"], message="Насколько надежны данные?")
        )
        assistant_payload = json.loads(assistant_response.body)["data"]["assistant"]

        self.assertEqual(assistant_payload["recording_id"], recording["id"])
        self.assertEqual(assistant_payload["quality_analysis_id"], payload["id"])
        self.assertEqual(assistant_payload["provider"]["type"], "ollama")
        self.assertIn("BPM", assistant_payload["message"])
        self.assertEqual(len(fake_provider.calls), 1)
        self.assertIn("quality_result", fake_provider.calls[0]["user_prompt"])
        self.assertNotIn("raw_data", fake_provider.calls[0]["user_prompt"])

    def test_measurement_detail_routes_exist_without_old_recording_exports(self) -> None:
        routes = {(route.path, tuple(sorted(route.methods))) for route in main_module.app.routes if hasattr(route, "methods")}

        self.assertTrue(any(path == "/api/measurements" and "GET" in methods for path, methods in routes))
        self.assertTrue(any(path == "/api/measurements/{measurement_id}" and "GET" in methods for path, methods in routes))
        self.assertTrue(any(path == "/api/assistant/status" and "GET" in methods for path, methods in routes))
        self.assertTrue(any(path == "/api/assistant/chat" and "POST" in methods for path, methods in routes))
        self.assertFalse(
            any(
                path in {
                    "/api/measurements/{measurement_id}/samples",
                    "/api/measurements/{measurement_id}/export",
                }
                and ("GET" in methods or "DELETE" in methods)
                for path, methods in routes
            )
        )


class MigrationTests(unittest.TestCase):
    def test_create_schema_backfills_public_ids_and_preserves_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            database_url = f"sqlite:///{Path(tempdir) / 'legacy.db'}"
            engine = create_engine(database_url, future=True)
            started_at = "2026-04-28 11:51:20"
            legacy_confidence_column = "sensor_" + "confidence"
            with engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        CREATE TABLE users (
                            id INTEGER PRIMARY KEY,
                            name VARCHAR(255) NOT NULL,
                            age INTEGER,
                            sex VARCHAR(32),
                            created_at DATETIME NOT NULL,
                            updated_at DATETIME NOT NULL,
                            recordings_qty INTEGER NOT NULL DEFAULT 0
                        )
                        """
                    )
                )
                connection.execute(
                    text(
                        """
                        CREATE TABLE projects (
                            id INTEGER PRIMARY KEY,
                            title VARCHAR(255) NOT NULL,
                            created_at DATETIME NOT NULL,
                            updated_at DATETIME NOT NULL,
                            recordings_qty INTEGER NOT NULL DEFAULT 0
                        )
                        """
                    )
                )
                connection.execute(
                    text(
                        f"""
                        CREATE TABLE recordings (
                            id VARCHAR(36) PRIMARY KEY,
                            user_name VARCHAR(255),
                            user_id VARCHAR(255),
                            project_name VARCHAR(255),
                            project_id VARCHAR(255),
                            started_at DATETIME NOT NULL,
                            finished_at DATETIME,
                            duration_ms BIGINT,
                            bpm FLOAT,
                            spo2 FLOAT,
                            status VARCHAR(32) NOT NULL,
                            signal_type VARCHAR(16) NOT NULL,
                            sample_rate FLOAT,
                            created_at DATETIME NOT NULL,
                            updated_at DATETIME NOT NULL,
                            signal_quality JSON,
                            sensor_temp FLOAT,
                            device_id VARCHAR(255),
                            perfusion_index FLOAT,
                            ratio FLOAT,
                            {legacy_confidence_column} FLOAT,
                            peak_count BIGINT
                        )
                        """
                    )
                )
                connection.execute(
                    text(
                        """
                        CREATE TABLE recordings_samples (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            recording_id VARCHAR(36) NOT NULL,
                            sample_index BIGINT NOT NULL,
                            raw_data JSON NOT NULL,
                            created_at DATETIME NOT NULL
                        )
                        """
                    )
                )
                connection.execute(
                    text("INSERT INTO users VALUES (1, 'Alice', 30, 'female', :ts, :ts, 0)"),
                    {"ts": started_at},
                )
                connection.execute(
                    text("INSERT INTO projects VALUES (1, 'Study', :ts, :ts, 0)"),
                    {"ts": started_at},
                )
                connection.execute(
                    text(
                        """
                        INSERT INTO recordings VALUES (
                            'legacy-rec', 'Alice', '1', 'Study', '1', :ts, NULL, NULL,
                            NULL, NULL, 'running', 'IR+R', 25, :ts, :ts, NULL, 31,
                            'sim-legacy', NULL, NULL, NULL, NULL
                        )
                        """
                    ),
                    {"ts": started_at},
                )
                connection.execute(
                    text(
                        "INSERT INTO recordings_samples (recording_id, sample_index, raw_data, created_at) "
                        "VALUES ('legacy-rec', 0, '{\"ir\": 1, \"red\": 2}', :ts)"
                    ),
                    {"ts": started_at},
                )
            engine.dispose()

            repository = RecordingRepository(database_url)
            repository.create_schema()

            user = repository.get_user("usr_1")
            project = repository.get_project("prj_1")
            recording = repository.get_recording("rec_legacy-rec")
            samples = repository.list_recording_samples("rec_legacy-rec", limit=None, offset=0)
            engine = create_engine(database_url, future=True)
            columns = {column["name"] for column in inspect(engine).get_columns("quality_analyses")}
            measurement_columns = {column["name"] for column in inspect(engine).get_columns("measurements")}
            recording_columns = {column["name"] for column in inspect(engine).get_columns("recordings")}
            engine.dispose()

            self.assertEqual(user["public_id"], "usr_1")
            self.assertEqual(project["public_id"], "prj_1")
            self.assertEqual(recording["public_id"], "rec_legacy-rec")
            self.assertEqual(recording["measurement_id"], "legacy-rec")
            self.assertEqual(recording["user_id"], "usr_1")
            self.assertEqual(recording["project_id"], "prj_1")
            self.assertEqual(len(samples), 1)
            self.assertIn("features", columns)
            self.assertNotIn(legacy_confidence_column, measurement_columns)
            self.assertNotIn(legacy_confidence_column, recording_columns)
            repository.dispose()


if __name__ == "__main__":
    unittest.main()

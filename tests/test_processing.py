import json
import math
import asyncio
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException
from sqlalchemy import create_engine, text

from app import main as main_module
from app.core.config import AppConfig
from app.main import _api_success, _datetime_for_api, _measurement_for_api, _metrics_for_api
from app.measurements import RecordingMetadata
from app.processing.metrics import VitalSignsCalculator
from app.processing.service import PPGProcessingService
from app.realtime.store import MetricsStore
from app.realtime.websocket import WebSocketController
from app.schemas.measurements import MeasurementStartRequest, ProjectPatchRequest, UserPatchRequest
from app.storage.recording_repository import RecordingRepository


def _synthetic_payload(
    fs: int = 25,
    seconds: int = 18,
    bpm: float = 72.0,
    measurement_id: str | None = None,
    device_id: str = "A0:B7:65:12:34:56",
) -> str:
    count = fs * seconds
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
    samples = [{"ir": float(i), "r": float(r)} for i, r in zip(ir, red, strict=True)]

    device = {
        "id": device_id,
        "temp": 31.75,
        "fs": fs,
        "samples": samples,
    }
    if measurement_id is not None:
        device["measurement_id"] = measurement_id

    return json.dumps(
        {
            "device": device
        }
    )


def _synthetic_payload_with_amplitude(ir_ac: float, red_ac: float, fs: int = 25, seconds: int = 18) -> str:
    count = fs * seconds
    t = np.arange(count) / fs
    freq = 72.0 / 60.0
    ir_dc = 84_000.0
    red_dc = 53_000.0
    ir = ir_dc + ir_ac * np.sin(2 * math.pi * freq * t)
    red = red_dc + red_ac * np.sin(2 * math.pi * freq * t)
    samples = [{"ir": float(i), "r": float(r)} for i, r in zip(ir, red, strict=True)]

    return json.dumps(
        {
            "device": {
                "id": "F4:65:0B:55:2E:80",
                "temp": 28.0,
                "fs": fs,
                "samples": samples,
            }
        }
    )


class ProcessingTests(unittest.TestCase):
    def test_processing_estimates_bpm_and_spo2(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())

        metrics = service.process_json(_synthetic_payload())

        self.assertIsNotNone(metrics.bpm)
        assert metrics.bpm is not None
        self.assertLessEqual(abs(metrics.bpm - 72.0), 4.0)
        self.assertIsNotNone(metrics.spo2)
        assert metrics.spo2 is not None
        self.assertGreaterEqual(metrics.spo2, 90.0)
        self.assertLessEqual(metrics.spo2, 100.0)
        self.assertGreaterEqual(metrics.sensor_confidence, 0.7)
        self.assertLessEqual(metrics.sensor_confidence, 1.0)
        self.assertIn(metrics.signal_quality.level, {"medium", "high"})

    def test_processing_warms_up_before_minimum_window(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())

        metrics = service.process_json(_synthetic_payload(seconds=2))

        self.assertIsNone(metrics.bpm)
        self.assertIsNone(metrics.spo2)
        self.assertGreaterEqual(metrics.sensor_confidence, 0.0)
        self.assertLessEqual(metrics.sensor_confidence, 0.2)
        self.assertEqual(metrics.signal_quality.level, "warming_up")

    def test_processing_waits_for_stable_window_before_reporting_spo2(self) -> None:
        service = PPGProcessingService(AppConfig(stable_spo2_window_seconds=15.0), MetricsStore())

        metrics = service.process_json(_synthetic_payload(seconds=10))

        self.assertIsNotNone(metrics.bpm)
        self.assertIsNone(metrics.spo2)
        self.assertIsNone(metrics.ratio)
        self.assertEqual(metrics.signal_quality.reason, "SpO2 not reported: waiting for stable RMS window")

    def test_processing_cuts_spo2_warmup_from_rms_window(self) -> None:
        fs = 25
        seconds = 12
        count = fs * seconds
        t = np.arange(count) / fs
        pulse = np.sin(2 * math.pi * 72.0 / 60.0 * t)
        ir_dc = 84_000.0
        red_dc = 53_000.0
        ir_ac = 1_500.0
        normal_ratio = 0.55
        red_ac = normal_ratio * ir_ac * red_dc / ir_dc

        filtered_ir = ir_ac * pulse
        filtered_red = red_ac * pulse
        filtered_red[t < 2.0] = 4.0 * red_ac * pulse[t < 2.0]
        raw_ir = ir_dc + filtered_ir
        raw_red = red_dc + filtered_red

        cut_result = VitalSignsCalculator(
            AppConfig(stable_spo2_window_seconds=8.0, spo2_warmup_cut_seconds=2.0)
        ).calculate(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            filtered_red=filtered_red,
            fs=fs,
        )
        no_cut_result = VitalSignsCalculator(
            AppConfig(stable_spo2_window_seconds=8.0, spo2_warmup_cut_seconds=0.0)
        ).calculate(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            filtered_red=filtered_red,
            fs=fs,
        )

        self.assertIsNotNone(cut_result.ratio)
        self.assertIsNotNone(no_cut_result.ratio)
        assert cut_result.ratio is not None
        assert no_cut_result.ratio is not None
        self.assertLessEqual(abs(cut_result.ratio - normal_ratio), 0.05)
        self.assertGreater(no_cut_result.ratio, cut_result.ratio + 0.2)

    def test_processing_rejects_no_finger_like_exposed_signal(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())

        metrics = service.process_json(_synthetic_payload_with_amplitude(ir_ac=34_000.0, red_ac=24_000.0))

        self.assertIsNone(metrics.bpm)
        self.assertIsNone(metrics.spo2)
        self.assertIsNone(metrics.ratio)
        self.assertEqual(metrics.sensor_confidence, 0.0)
        self.assertEqual(metrics.signal_quality.level, "no_contact")
        self.assertEqual(metrics.signal_quality.reason, "finger not detected: optical signal is unstable or exposed")

    def test_processing_keeps_bpm_but_rejects_spo2_when_perfusion_is_too_low(self) -> None:
        config = AppConfig(min_spo2_perfusion_index=0.4)
        service = PPGProcessingService(config, MetricsStore())

        metrics = service.process_json(_synthetic_payload_with_amplitude(ir_ac=90.0, red_ac=70.0))

        self.assertIsNotNone(metrics.bpm)
        self.assertIsNone(metrics.spo2)
        self.assertIsNotNone(metrics.ratio)
        self.assertEqual(
            metrics.signal_quality.reason,
            "SpO2 not reported: perfusion index is too low for oxygen estimation",
        )

    def test_measurement_session_returns_final_result(self) -> None:
        config = AppConfig(measurement_duration_seconds=15.0)
        service = PPGProcessingService(config, MetricsStore())
        service.start_measurement("A0:B7:65:12:34:56")

        service.process_json(_synthetic_payload(seconds=18))
        measurement = service.get_measurement("A0:B7:65:12:34:56")

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertEqual(measurement.status, "completed")
        self.assertEqual(measurement.duration_seconds, 15.0)
        self.assertEqual(measurement.samples_collected, 375)
        self.assertIsNotNone(measurement.result)
        assert measurement.result is not None
        self.assertIsNotNone(measurement.result.bpm)
        self.assertIsNotNone(measurement.result.spo2)

    def test_measurement_ignores_samples_for_different_measurement_id(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())
        service.start_measurement("A0:B7:65:12:34:56")

        service.process_json(_synthetic_payload(measurement_id="different-measurement"))
        measurement = service.get_measurement("A0:B7:65:12:34:56")

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertEqual(measurement.status, "running")
        self.assertEqual(measurement.samples_collected, 0)

    def test_measurement_accepts_matching_measurement_id(self) -> None:
        service = PPGProcessingService(AppConfig(measurement_duration_seconds=15.0), MetricsStore())
        started = service.start_measurement("A0:B7:65:12:34:56")
        assert started.id is not None

        service.process_json(_synthetic_payload(measurement_id=started.id))
        measurement = service.get_measurement("A0:B7:65:12:34:56")

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertEqual(measurement.status, "completed")
        self.assertEqual(measurement.samples_collected, 375)

    def test_measurement_can_stop_active_recording_by_device_id(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())
        service.start_measurement("A0:B7:65:12:34:56")
        service.process_json(_synthetic_payload(seconds=2))

        stopped = service.stop_recording_for_device("A0:B7:65:12:34:56")

        self.assertIsNotNone(stopped)
        assert stopped is not None
        self.assertEqual(stopped.status, "stopped")

    def test_live_sample_payload_contains_measurement_samples_and_metrics(self) -> None:
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
        raw_samples = json.loads(raw_payload)["device"]["samples"]

        self.assertIsNotNone(live_payload)
        assert live_payload is not None
        self.assertEqual(live_payload["type"], "measurement_sample_batch")
        self.assertEqual(live_payload["measurement_id"], started.id)
        self.assertEqual(live_payload["device_id"], "A0:B7:65:12:34:56")
        self.assertEqual(live_payload["fs"], 25)
        self.assertEqual(live_payload["sample_count"], 25)
        self.assertEqual(len(live_payload["samples"]), 25)
        self.assertEqual(live_payload["samples"], processed.processed_samples)
        self.assertNotEqual(live_payload["samples"][0]["ir"], raw_samples[0]["ir"])
        self.assertNotEqual(live_payload["samples"][0]["r"], raw_samples[0]["r"])
        self.assertEqual(live_payload["metrics"]["device_id"], "A0:B7:65:12:34:56")
        self.assertIn("measured_at", live_payload["metrics"]["time"])
        self.assertEqual(live_payload["metrics"]["sample_rate"], 25)
        self.assertNotIn("timestamp", live_payload["metrics"])
        self.assertNotIn("fs", live_payload["metrics"])

    def test_api_metrics_use_v31_shape(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())

        metrics = service.process_json(_synthetic_payload())
        payload = _metrics_for_api(metrics)

        self.assertEqual(payload["device_id"], "A0:B7:65:12:34:56")
        self.assertIn("measured_at", payload["time"])
        self.assertEqual(payload["sample_rate"], 25)
        self.assertEqual(payload["sensor_temp"], 31.75)
        self.assertNotIn("timestamp", payload)
        self.assertNotIn("fs", payload)
        self.assertNotIn("temperature", payload)
        assert isinstance(payload["signal_quality"], dict)
        self.assertIn("samples_in_window", payload["signal_quality"])

    def test_api_measurement_uses_v31_shape(self) -> None:
        service = PPGProcessingService(AppConfig(measurement_duration_seconds=15.0), MetricsStore())
        started = service.start_recording(
            duration_seconds=15.0,
            metadata=RecordingMetadata(
                user_id="1",
                project_id="1",
            ),
            device_id="sim-device-001",
        )
        assert started.id is not None

        service.process_json(
            _synthetic_payload(
                seconds=18,
                measurement_id=started.id,
                device_id="sim-device-001",
            )
        )
        measurement = service.get_measurement("sim-device-001")
        self.assertIsNotNone(measurement)
        assert measurement is not None

        payload = _measurement_for_api(measurement)

        self.assertEqual(payload["id"], started.id)
        self.assertTrue(payload["is_simulated"])
        self.assertEqual(payload["user_id"], "1")
        self.assertEqual(payload["project_id"], "1")
        self.assertEqual(payload["device_id"], "sim-device-001")
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["channels"], ["ir", "red"])
        self.assertIn("started_at", payload["time"])
        self.assertIn("finished_at", payload["time"])
        self.assertIn("duration_ms", payload["time"])
        self.assertEqual(payload["sample_rate"], 25)
        self.assertEqual(payload["sensor_temp"], 31.75)
        self.assertNotIn("started_at", payload)
        self.assertNotIn("finished_at", payload)
        self.assertNotIn("duration_seconds", payload)
        assert isinstance(payload["signal_quality"], dict)
        self.assertIn("perfusion_index", payload["signal_quality"])

    def test_api_datetime_format_is_utc_iso_z(self) -> None:
        value = datetime(2026, 4, 28, 11, 51, 20, 778832, tzinfo=UTC)

        self.assertEqual(_datetime_for_api(value), "2026-04-28T11:51:20.778832Z")
        self.assertEqual(
            _datetime_for_api("2026-04-28T14:51:20.778832+03:00"),
            "2026-04-28T11:51:20.778832Z",
        )

    def test_api_success_formats_nested_datetimes(self) -> None:
        response = _api_success(
            {
                "created_at": datetime(2026, 4, 28, 11, 51, 20, 778832, tzinfo=UTC),
                "items": [
                    {"updated_at": datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)},
                ],
            }
        )
        payload = json.loads(response.body)

        self.assertEqual(payload["data"]["created_at"], "2026-04-28T11:51:20.778832Z")
        self.assertEqual(payload["data"]["items"][0]["updated_at"], "2026-04-28T12:00:00Z")
        self.assertTrue(payload["$meta"]["time"]["iso"].endswith("Z"))
        self.assertNotIn("ts", payload["$meta"]["time"])


class WebSocketControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_start_waits_for_start_ack(self) -> None:
        controller = WebSocketController(object(), ack_timeout_seconds=0.05)

        class AckingWebSocket:
            async def send_json(self, payload: dict[str, Any]) -> None:
                controller._handle_device_control_message(
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
            measurement_id="measurement-1",
            duration_seconds=10.0,
        )

        self.assertTrue(sent)

    async def test_send_start_fails_when_start_ack_is_missing(self) -> None:
        controller = WebSocketController(object(), ack_timeout_seconds=0.01)

        class SilentWebSocket:
            async def send_json(self, _payload: dict[str, Any]) -> None:
                return None

        websocket = SilentWebSocket()
        controller._connections["sim-test"] = websocket

        sent = await controller.send_start(
            device_id="sim-test",
            measurement_id="measurement-1",
            duration_seconds=10.0,
        )

        self.assertFalse(sent)
        self.assertNotIn("sim-test", controller._connections)


class ApiMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_repository = main_module.recording_repository
        self._old_processing_service = main_module.processing_service
        self._old_ws_controller = main_module.ws_controller
        self._old_metrics_store = main_module.metrics_store
        self._tempdir = tempfile.TemporaryDirectory()
        database_url = f"sqlite:///{Path(self._tempdir.name) / 'test.db'}"
        self.repository = RecordingRepository(database_url)
        self.repository.create_schema()
        self.metrics_store = MetricsStore()
        self.processing_service = PPGProcessingService(
            AppConfig(database_url=database_url),
            self.metrics_store,
            self.repository,
        )
        main_module.recording_repository = self.repository
        main_module.processing_service = self.processing_service
        main_module.metrics_store = self.metrics_store
        main_module.ws_controller = WebSocketController(
            self.processing_service,
            metrics_formatter=main_module._metrics_for_api,
            ack_timeout_seconds=0.01,
        )

    def tearDown(self) -> None:
        main_module.recording_repository = self._old_repository
        main_module.processing_service = self._old_processing_service
        main_module.ws_controller = self._old_ws_controller
        main_module.metrics_store = self._old_metrics_store
        self.repository.dispose()
        self._tempdir.cleanup()

    def test_patch_and_delete_user(self) -> None:
        created = self.repository.create_user({"name": "Alice", "age": 30, "sex": "female"})

        response = main_module.patch_api_user(created["id"], UserPatchRequest(age=31))
        payload = json.loads(response.body)

        self.assertEqual(payload["data"]["user"]["name"], "Alice")
        self.assertEqual(payload["data"]["user"]["age"], 31)

        response = main_module.delete_api_user(created["id"])
        payload = json.loads(response.body)

        self.assertTrue(payload["data"]["deleted"])
        self.assertIsNone(self.repository.get_user(created["id"]))

    def test_patch_and_delete_project(self) -> None:
        created = self.repository.create_project({"title": "Old", "description": "draft"})

        response = main_module.patch_api_project(
            created["id"],
            ProjectPatchRequest(title="New", description=None),
        )
        payload = json.loads(response.body)

        self.assertEqual(payload["data"]["project"]["title"], "New")
        self.assertIsNone(payload["data"]["project"]["description"])

        response = main_module.delete_api_project(created["id"])
        payload = json.loads(response.body)

        self.assertTrue(payload["data"]["deleted"])
        self.assertIsNone(self.repository.get_project(created["id"]))

    def test_create_schema_adds_project_description_to_existing_table(self) -> None:
        database_url = f"sqlite:///{Path(self._tempdir.name) / 'legacy.db'}"
        engine = create_engine(database_url, future=True)
        with engine.begin() as connection:
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
        engine.dispose()

        repository = RecordingRepository(database_url)
        repository.create_schema()
        project = repository.create_project({"title": "Migrated", "description": "exists"})

        self.assertEqual(project["description"], "exists")
        self.assertEqual(repository.list_projects()[0]["description"], "exists")
        repository.dispose()

    def test_delete_measurement_removes_recording(self) -> None:
        started = self.processing_service.start_recording(
            duration_seconds=15.0,
            metadata=RecordingMetadata(user_id="1", project_id="1"),
            device_id="sim-delete-001",
        )
        assert started.id is not None

        response = asyncio.run(main_module.delete_api_measurement(started.id))
        payload = json.loads(response.body)

        self.assertTrue(payload["data"]["deleted"])
        self.assertEqual(payload["data"]["measurement_id"], started.id)
        self.assertIsNone(self.repository.get_recording(started.id))

    def test_delete_device_removes_cached_metrics(self) -> None:
        device_id = "sim-delete-device"
        self.processing_service.process_json(_synthetic_payload(device_id=device_id))
        self.assertIsNotNone(self.metrics_store.get(device_id))

        response = asyncio.run(main_module.delete_api_device(device_id))
        payload = json.loads(response.body)

        self.assertTrue(payload["data"]["deleted"])
        self.assertEqual(payload["data"]["device_id"], device_id)
        self.assertIsNone(self.metrics_store.get(device_id))

    def test_start_measurement_rejects_active_measurement_for_same_device(self) -> None:
        class AckingController:
            async def send_start(
                self,
                *,
                device_id: str,
                measurement_id: str,
                duration_seconds: float | None,
            ) -> bool:
                return True

        main_module.ws_controller = AckingController()
        body = MeasurementStartRequest(duration_s=15)
        asyncio.run(main_module.start_api_measurement("sim-active-device", body))

        with self.assertRaises(HTTPException) as context:
            asyncio.run(main_module.start_api_measurement("sim-active-device", body))

        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(str(context.exception.detail), "measurement is already running for device")

    def test_stop_measurement_rejects_inactive_measurement(self) -> None:
        started = self.processing_service.start_recording(
            duration_seconds=15.0,
            metadata=RecordingMetadata(user_id="1", project_id="1"),
            device_id="sim-stop-once",
        )
        assert started.id is not None

        first_response = asyncio.run(main_module.stop_api_measurement(started.id))
        first_payload = json.loads(first_response.body)
        self.assertEqual(first_payload["data"]["measurement"]["status"], "cancelled")

        with self.assertRaises(HTTPException) as context:
            asyncio.run(main_module.stop_api_measurement(started.id))

        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(str(context.exception.detail), "measurement is not active")

    def test_measurement_samples_endpoint_returns_processed_signal(self) -> None:
        started = self.processing_service.start_recording(
            duration_seconds=15.0,
            metadata=RecordingMetadata(user_id="1", project_id="1"),
            device_id="sim-samples-processed",
        )
        assert started.id is not None
        raw_payload = _synthetic_payload(
            seconds=2,
            measurement_id=started.id,
            device_id="sim-samples-processed",
        )
        raw_samples = json.loads(raw_payload)["device"]["samples"]

        self.processing_service.process_json(raw_payload)
        response = main_module.get_api_measurement_samples(started.id, limit=10, offset=0)
        payload = json.loads(response.body)
        samples = payload["data"]["samples"]

        self.assertEqual(len(samples), 10)
        self.assertEqual(set(samples[0]), {"ir", "r"})
        self.assertNotEqual(samples[0]["ir"], raw_samples[0]["ir"])
        self.assertNotEqual(samples[0]["r"], raw_samples[0]["r"])

        all_processed = self.processing_service.get_recording_processed_samples(
            started.id,
            limit=None,
            offset=0,
        )
        paged_response = main_module.get_api_measurement_samples(started.id, limit=5, offset=5)
        paged_payload = json.loads(paged_response.body)
        self.assertEqual(paged_payload["data"]["samples"], all_processed[5:10])


if __name__ == "__main__":
    unittest.main()

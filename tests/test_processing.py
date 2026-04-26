import json
import math
import unittest

import numpy as np

from app.config import AppConfig
from app.processing.service import PPGProcessingService
from app.state import MetricsStore


def _synthetic_payload(
    fs: int = 25,
    seconds: int = 18,
    bpm: float = 72.0,
    recording_id: str | None = None,
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
        "id": "A0:B7:65:12:34:56",
        "temp": 31.75,
        "fs": fs,
        "samples": samples,
    }
    if recording_id is not None:
        device["recording_id"] = recording_id

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

    def test_measurement_ignores_samples_for_different_recording_id(self) -> None:
        service = PPGProcessingService(AppConfig(), MetricsStore())
        service.start_measurement("A0:B7:65:12:34:56")

        service.process_json(_synthetic_payload(recording_id="different-recording"))
        measurement = service.get_measurement("A0:B7:65:12:34:56")

        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertEqual(measurement.status, "running")
        self.assertEqual(measurement.samples_collected, 0)

    def test_measurement_accepts_matching_recording_id(self) -> None:
        service = PPGProcessingService(AppConfig(measurement_duration_seconds=15.0), MetricsStore())
        started = service.start_measurement("A0:B7:65:12:34:56")
        assert started.id is not None

        service.process_json(_synthetic_payload(recording_id=started.id))
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


if __name__ == "__main__":
    unittest.main()

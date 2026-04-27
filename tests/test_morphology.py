import unittest

import numpy as np

from app.processing.morphology import DEFAULT_TEMPLATE_LENGTH, analyze_waveform_morphology


def _pulse_template(length: int, peak_fraction: float = 0.28, decay_power: float = 0.9) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length, endpoint=False)
    rise = np.power(np.clip(x / peak_fraction, 0.0, 1.0), 1.7)
    decay_x = np.clip((x - peak_fraction) / (1.0 - peak_fraction), 0.0, 1.0)
    decay = np.power(1.0 - decay_x, decay_power)
    pulse = np.where(x <= peak_fraction, rise, decay)
    pulse[0] = 0.0
    pulse[-1] = 0.0
    return pulse


def _synthetic_ppg(
    *,
    fs: int = 50,
    cycle_lengths: list[int] | None = None,
    amplitude: float = 1000.0,
    noise_std: float = 0.0,
    distorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if cycle_lengths is None:
        cycle_lengths = [42] * 14

    rng = np.random.default_rng(123)
    cycles: list[np.ndarray] = []
    peaks: list[int] = []
    offset = 0
    for index, length in enumerate(cycle_lengths):
        if distorted and index % 3 == 1:
            x = np.linspace(0.0, 1.0, length, endpoint=False)
            peak_fraction = 0.78
            rise = np.power(np.clip(x / peak_fraction, 0.0, 1.0), 0.55)
            decay_x = np.clip((x - peak_fraction) / (1.0 - peak_fraction), 0.0, 1.0)
            decay = np.power(1.0 - decay_x, 2.8)
            template = np.where(x <= peak_fraction, rise, decay)
            template[0] = 0.0
            template[-1] = 0.0
        elif distorted and index % 3 == 2:
            x = np.linspace(0.0, 1.0, length, endpoint=False)
            template = np.exp(-np.square((x - 0.22) / 0.06))
            template += 0.85 * np.exp(-np.square((x - 0.72) / 0.08))
            template -= 0.45 * np.exp(-np.square((x - 0.48) / 0.07))
            template = (template - float(np.min(template))) / float(np.ptp(template))
            template[0] = 0.0
            template[-1] = 0.0
        else:
            template = _pulse_template(length)
        cycles.append(template)
        peaks.append(offset + int(np.argmax(template)))
        offset += length

    filtered = amplitude * np.concatenate(cycles)
    if noise_std > 0:
        filtered = filtered + rng.normal(0.0, noise_std * amplitude, filtered.size)
    raw = 84_000.0 + filtered
    return raw, filtered, np.asarray(peaks, dtype=int), float(fs)


class WaveformMorphologyTests(unittest.TestCase):
    def test_clean_regular_ppg_has_high_shape_score(self) -> None:
        raw, filtered, peaks, fs = _synthetic_ppg()

        result = analyze_waveform_morphology(raw_ir=raw, filtered_ir=filtered, peaks=peaks, fs=fs)

        self.assertEqual(result.shape_quality, "stable")
        self.assertGreaterEqual(result.valid_pulse_count, 10)
        assert result.shape_score is not None
        self.assertGreater(result.shape_score, 0.9)
        assert result.shape_similarity is not None
        self.assertGreater(result.shape_similarity, 0.95)

    def test_noisy_ppg_preserves_morphology_when_shape_is_consistent(self) -> None:
        raw, filtered, peaks, fs = _synthetic_ppg(noise_std=0.035)

        result = analyze_waveform_morphology(raw_ir=raw, filtered_ir=filtered, peaks=peaks, fs=fs)

        self.assertIn(result.shape_quality, {"stable", "moderately_stable"})
        assert result.shape_score is not None
        self.assertGreater(result.shape_score, 0.75)
        assert result.shape_similarity is not None
        self.assertGreater(result.shape_similarity, 0.85)

    def test_distorted_pulse_shapes_lower_shape_score(self) -> None:
        stable_raw, stable_filtered, stable_peaks, fs = _synthetic_ppg()
        distorted_raw, distorted_filtered, distorted_peaks, _ = _synthetic_ppg(distorted=True)

        stable = analyze_waveform_morphology(
            raw_ir=stable_raw,
            filtered_ir=stable_filtered,
            peaks=stable_peaks,
            fs=fs,
        )
        distorted = analyze_waveform_morphology(
            raw_ir=distorted_raw,
            filtered_ir=distorted_filtered,
            peaks=distorted_peaks,
            fs=fs,
        )

        assert stable.shape_score is not None
        assert distorted.shape_score is not None
        self.assertLess(distorted.shape_score, stable.shape_score - 0.15)
        self.assertLess(distorted.shape_score, 0.85)

    def test_insufficient_peaks_return_low_confidence_result(self) -> None:
        raw, filtered, peaks, fs = _synthetic_ppg(cycle_lengths=[42, 42])

        result = analyze_waveform_morphology(raw_ir=raw, filtered_ir=filtered, peaks=peaks, fs=fs)

        self.assertEqual(result.shape_quality, "insufficient_pulses")
        self.assertIsNone(result.shape_score)
        self.assertIsNone(result.average_pulse_template)

    def test_too_low_amplitude_is_reported(self) -> None:
        raw, filtered, peaks, fs = _synthetic_ppg(amplitude=1e-8)

        result = analyze_waveform_morphology(raw_ir=raw, filtered_ir=filtered, peaks=peaks, fs=fs)

        self.assertEqual(result.shape_quality, "low_amplitude")
        self.assertEqual(result.shape_score, 0.0)
        self.assertEqual(result.reason, "pulse amplitude is too low")

    def test_invalid_signal_handles_nan_without_crashing(self) -> None:
        raw, filtered, peaks, fs = _synthetic_ppg()
        filtered[5] = np.nan

        result = analyze_waveform_morphology(raw_ir=raw, filtered_ir=filtered, peaks=peaks, fs=fs)

        self.assertEqual(result.shape_quality, "invalid_signal")
        self.assertIsNone(result.shape_score)

    def test_variable_pulse_durations_increase_duration_variability(self) -> None:
        raw, filtered, peaks, fs = _synthetic_ppg(
            cycle_lengths=[34, 44, 52, 38, 48, 58, 36, 50, 42, 54, 40, 46]
        )

        result = analyze_waveform_morphology(raw_ir=raw, filtered_ir=filtered, peaks=peaks, fs=fs)

        assert result.duration_variability is not None
        self.assertGreater(result.duration_variability, 0.12)
        assert result.shape_score is not None
        self.assertLess(result.shape_score, 0.9)

    def test_average_pulse_template_has_fixed_length(self) -> None:
        raw, filtered, peaks, fs = _synthetic_ppg()

        result = analyze_waveform_morphology(raw_ir=raw, filtered_ir=filtered, peaks=peaks, fs=fs)

        self.assertIsNotNone(result.average_pulse_template)
        assert result.average_pulse_template is not None
        self.assertEqual(len(result.average_pulse_template), DEFAULT_TEMPLATE_LENGTH)


if __name__ == "__main__":
    unittest.main()

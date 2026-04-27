from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.models import WaveformMorphology


MIN_VALID_PULSES = 3
DEFAULT_TEMPLATE_LENGTH = 100
LOW_ABSOLUTE_AMPLITUDE = 1e-6


@dataclass(frozen=True)
class _PulseFeatures:
    amplitude: float
    duration_ms: float
    rise_time_ms: float
    decay_time_ms: float
    pulse_width_50_ms: float
    rise_slope: float
    decay_slope: float
    area: float
    symmetry_ratio: float
    normalized: np.ndarray


def analyze_waveform_morphology(
    *,
    raw_ir: Sequence[float] | np.ndarray,
    filtered_ir: Sequence[float] | np.ndarray,
    peaks: Sequence[int] | np.ndarray,
    fs: float,
    raw_red: Sequence[float] | np.ndarray | None = None,
    filtered_red: Sequence[float] | np.ndarray | None = None,
    min_bpm: float = 40.0,
    max_bpm: float = 180.0,
    template_length: int = DEFAULT_TEMPLATE_LENGTH,
) -> WaveformMorphology:
    del raw_red, filtered_red

    try:
        raw = _as_float_array(raw_ir)
        filtered = _as_float_array(filtered_ir)
    except (TypeError, ValueError):
        return _empty_result("invalid_signal", "invalid signal input")

    if not _valid_signal_inputs(raw, filtered, fs, template_length) or not _valid_bpm_limits(
        min_bpm,
        max_bpm,
    ):
        return _empty_result("invalid_signal", "invalid signal input")

    try:
        peak_indexes = _clean_peaks(peaks, filtered.size)
    except (TypeError, ValueError):
        return _empty_result("invalid_signal", "invalid signal input")
    if _low_amplitude_signal(filtered):
        return _empty_result("low_amplitude", "pulse amplitude is too low", shape_score=0.0)
    if peak_indexes.size < 3:
        return _empty_result("insufficient_pulses", "not enough valid pulse segments")

    foot_points = _interpeak_foot_points(filtered, peak_indexes)
    if foot_points.size < 2:
        return _empty_result("insufficient_pulses", "not enough valid pulse segments")

    min_duration_seconds = 60.0 / max_bpm
    max_duration_seconds = 60.0 / min_bpm
    min_amplitude = max(LOW_ABSOLUTE_AMPLITUDE, float(np.std(filtered)) * 0.10)
    features: list[_PulseFeatures] = []
    rejected_for_low_amplitude = 0

    for peak_position in range(1, peak_indexes.size - 1):
        start = int(foot_points[peak_position - 1])
        end = int(foot_points[peak_position])
        peak = int(peak_indexes[peak_position])
        duration_seconds = (end - start) / fs
        if duration_seconds < min_duration_seconds or duration_seconds > max_duration_seconds:
            continue

        pulse = _pulse_features(
            filtered=filtered,
            start=start,
            peak=peak,
            end=end,
            fs=fs,
            min_amplitude=min_amplitude,
            template_length=template_length,
        )
        if pulse is None:
            if start < peak < end:
                amplitude = float(filtered[peak] - filtered[start])
                if amplitude < min_amplitude:
                    rejected_for_low_amplitude += 1
            continue
        features.append(pulse)

    if not features:
        if rejected_for_low_amplitude:
            return _empty_result("low_amplitude", "pulse amplitude is too low", shape_score=0.0)
        return _empty_result("insufficient_pulses", "not enough valid pulse segments")

    features = _reject_obvious_amplitude_outliers(features)
    if len(features) < MIN_VALID_PULSES:
        return WaveformMorphology(
            enabled=True,
            valid_pulse_count=len(features),
            shape_quality="insufficient_pulses",
            reason="not enough valid pulse segments",
        )

    normalized_pulses = np.vstack([feature.normalized for feature in features])
    average_template = np.mean(normalized_pulses, axis=0)
    correlations = [
        correlation
        for pulse in normalized_pulses
        if (correlation := _correlation(pulse, average_template)) is not None
    ]
    if not correlations:
        return WaveformMorphology(
            enabled=True,
            valid_pulse_count=len(features),
            shape_quality="irregular_shape",
            reason="pulse shapes have low correlation",
            shape_score=0.0,
        )

    amplitudes = np.asarray([feature.amplitude for feature in features], dtype=float)
    durations = np.asarray([feature.duration_ms for feature in features], dtype=float)
    rise_times = np.asarray([feature.rise_time_ms for feature in features], dtype=float)
    widths = np.asarray([feature.pulse_width_50_ms for feature in features], dtype=float)
    shape_similarity = float(np.median(correlations))
    amplitude_variability = _coefficient_of_variation(amplitudes)
    duration_variability = _coefficient_of_variation(durations)
    rise_variability = _coefficient_of_variation(rise_times)
    width_variability = _coefficient_of_variation(widths)
    shape_score = _shape_score(
        shape_similarity=shape_similarity,
        amplitude_variability=amplitude_variability,
        duration_variability=duration_variability,
        rise_variability=rise_variability,
        width_variability=width_variability,
        valid_pulse_count=len(features),
    )
    shape_quality, reason = _shape_quality(
        score=shape_score,
        shape_similarity=shape_similarity,
        amplitude_variability=amplitude_variability,
        duration_variability=duration_variability,
    )

    return WaveformMorphology(
        enabled=True,
        valid_pulse_count=len(features),
        pulse_amplitude=_rounded_median(amplitudes),
        pulse_duration_ms=_rounded_median(durations),
        rise_time_ms=_rounded_median(rise_times),
        decay_time_ms=_rounded_median([feature.decay_time_ms for feature in features]),
        pulse_width_50_ms=_rounded_median(widths),
        rise_slope=_rounded_median([feature.rise_slope for feature in features]),
        decay_slope=_rounded_median([feature.decay_slope for feature in features]),
        area=_rounded_median([feature.area for feature in features], ndigits=4),
        symmetry_ratio=_rounded_median([feature.symmetry_ratio for feature in features], ndigits=4),
        shape_similarity=_finite_round(shape_similarity),
        amplitude_variability=_finite_round(amplitude_variability),
        duration_variability=_finite_round(duration_variability),
        morphology_variability=_finite_round(1.0 - shape_similarity),
        shape_score=_finite_round(shape_score),
        shape_quality=shape_quality,
        reason=reason,
        average_pulse_template=_rounded_template(average_template),
    )


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _valid_signal_inputs(raw: np.ndarray, filtered: np.ndarray, fs: float, template_length: int) -> bool:
    return (
        raw.size >= 3
        and raw.size == filtered.size
        and template_length >= 3
        and np.isfinite(fs)
        and fs > 0
        and np.all(np.isfinite(raw))
        and np.all(np.isfinite(filtered))
    )


def _valid_bpm_limits(min_bpm: float, max_bpm: float) -> bool:
    return (
        np.isfinite(min_bpm)
        and np.isfinite(max_bpm)
        and min_bpm > 0
        and max_bpm > 0
        and min_bpm <= max_bpm
    )


def _clean_peaks(peaks: Sequence[int] | np.ndarray, signal_size: int) -> np.ndarray:
    peak_array = np.asarray(peaks).reshape(-1)
    if peak_array.size == 0:
        return np.empty(0, dtype=int)
    peak_array = peak_array[np.isfinite(peak_array)]
    peak_array = peak_array.astype(int, copy=False)
    peak_array = peak_array[(peak_array > 0) & (peak_array < signal_size - 1)]
    return np.unique(peak_array)


def _low_amplitude_signal(filtered: np.ndarray) -> bool:
    return float(np.ptp(filtered)) <= LOW_ABSOLUTE_AMPLITUDE


def _interpeak_foot_points(filtered: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    foot_points: list[int] = []
    for left_peak, right_peak in zip(peaks[:-1], peaks[1:], strict=True):
        start = int(left_peak) + 1
        end = int(right_peak)
        if end <= start:
            continue
        interval = filtered[start:end]
        foot_points.append(start + int(np.argmin(interval)))
    return np.asarray(foot_points, dtype=int)


def _pulse_features(
    *,
    filtered: np.ndarray,
    start: int,
    peak: int,
    end: int,
    fs: float,
    min_amplitude: float,
    template_length: int,
) -> _PulseFeatures | None:
    if not start < peak < end:
        return None

    segment = filtered[start : end + 1]
    if segment.size < 3 or not np.all(np.isfinite(segment)):
        return None

    peak_offset = int(peak - start)
    local_peak_offset = int(np.argmax(segment))
    if abs(local_peak_offset - peak_offset) <= max(1, int(round(fs * 0.12))):
        peak_offset = local_peak_offset

    if peak_offset <= 0 or peak_offset >= segment.size - 1:
        return None

    start_value = float(segment[0])
    peak_value = float(segment[peak_offset])
    amplitude = peak_value - start_value
    if amplitude < min_amplitude:
        return None

    duration_ms = (segment.size - 1) / fs * 1000.0
    rise_time_ms = peak_offset / fs * 1000.0
    decay_time_ms = (segment.size - 1 - peak_offset) / fs * 1000.0
    if min(rise_time_ms, decay_time_ms) <= 0:
        return None

    width_50_ms = _pulse_width_50_ms(segment, peak_offset, fs, start_value, amplitude)
    if width_50_ms is None:
        return None

    normalized = _normalized_resample(segment, template_length)
    if normalized is None:
        return None

    area = float(np.trapezoid(normalized, dx=1.0 / (template_length - 1)))
    return _PulseFeatures(
        amplitude=float(amplitude),
        duration_ms=float(duration_ms),
        rise_time_ms=float(rise_time_ms),
        decay_time_ms=float(decay_time_ms),
        pulse_width_50_ms=float(width_50_ms),
        rise_slope=float(amplitude / rise_time_ms),
        decay_slope=float(amplitude / decay_time_ms),
        area=area,
        symmetry_ratio=float(rise_time_ms / duration_ms),
        normalized=normalized,
    )


def _pulse_width_50_ms(
    segment: np.ndarray,
    peak_offset: int,
    fs: float,
    baseline: float,
    amplitude: float,
) -> float | None:
    level = baseline + amplitude * 0.5
    left_crossing = _first_upward_crossing(segment[: peak_offset + 1], level)
    right_crossing = _first_downward_crossing(segment[peak_offset:], level)
    if left_crossing is None or right_crossing is None:
        return None

    right_crossing += peak_offset
    if right_crossing <= left_crossing:
        return None
    return (right_crossing - left_crossing) / fs * 1000.0


def _first_upward_crossing(values: np.ndarray, level: float) -> float | None:
    indexes = np.flatnonzero(values >= level)
    if indexes.size == 0:
        return None
    index = int(indexes[0])
    if index == 0:
        return 0.0
    return _interpolated_index(index - 1, values[index - 1], index, values[index], level)


def _first_downward_crossing(values: np.ndarray, level: float) -> float | None:
    indexes = np.flatnonzero(values <= level)
    indexes = indexes[indexes > 0]
    if indexes.size == 0:
        return None
    index = int(indexes[0])
    return _interpolated_index(index - 1, values[index - 1], index, values[index], level)


def _interpolated_index(
    left_index: int,
    left_value: float,
    right_index: int,
    right_value: float,
    level: float,
) -> float:
    span = float(right_value - left_value)
    if abs(span) <= 1e-12:
        return float(right_index)
    fraction = _clamp((level - float(left_value)) / span)
    return float(left_index) + fraction * float(right_index - left_index)


def _normalized_resample(segment: np.ndarray, template_length: int) -> np.ndarray | None:
    source_x = np.arange(segment.size, dtype=float)
    target_x = np.linspace(0.0, float(segment.size - 1), template_length)
    resampled = np.interp(target_x, source_x, segment)
    minimum = float(np.min(resampled))
    span = float(np.ptp(resampled))
    if span <= LOW_ABSOLUTE_AMPLITUDE:
        return None
    return (resampled - minimum) / span


def _reject_obvious_amplitude_outliers(features: list[_PulseFeatures]) -> list[_PulseFeatures]:
    if len(features) < 4:
        return features
    amplitudes = np.asarray([feature.amplitude for feature in features], dtype=float)
    median_amplitude = float(np.median(amplitudes))
    if median_amplitude <= 0:
        return []
    keep = (amplitudes >= median_amplitude * 0.20) & (amplitudes <= median_amplitude * 5.0)
    return [feature for feature, should_keep in zip(features, keep, strict=True) if bool(should_keep)]


def _correlation(pulse: np.ndarray, template: np.ndarray) -> float | None:
    pulse_centered = pulse - float(np.mean(pulse))
    template_centered = template - float(np.mean(template))
    denominator = float(np.linalg.norm(pulse_centered) * np.linalg.norm(template_centered))
    if denominator <= 1e-12:
        return None
    return _clamp(float(np.dot(pulse_centered, template_centered) / denominator))


def _coefficient_of_variation(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 1.0
    mean = abs(float(np.mean(array)))
    if mean <= 1e-12:
        return 1.0
    return float(np.std(array) / mean)


def _shape_score(
    *,
    shape_similarity: float,
    amplitude_variability: float,
    duration_variability: float,
    rise_variability: float,
    width_variability: float,
    valid_pulse_count: int,
) -> float:
    amplitude_stability = _clamp(1.0 - amplitude_variability / 0.35)
    duration_stability = _clamp(1.0 - duration_variability / 0.25)
    rise_stability = _clamp(1.0 - rise_variability / 0.35)
    width_stability = _clamp(1.0 - width_variability / 0.40)
    feature_stability = (rise_stability + width_stability) / 2.0
    valid_pulse_count_score = _clamp((valid_pulse_count - 2.0) / 6.0)
    return _clamp(
        0.45 * shape_similarity
        + 0.20 * amplitude_stability
        + 0.20 * duration_stability
        + 0.10 * valid_pulse_count_score
        + 0.05 * feature_stability
    )


def _shape_quality(
    *,
    score: float,
    shape_similarity: float,
    amplitude_variability: float,
    duration_variability: float,
) -> tuple[str, str | None]:
    reason = _dominant_reason(shape_similarity, amplitude_variability, duration_variability)
    if shape_similarity < 0.55:
        return "irregular_shape", "pulse shapes have low correlation"
    if score >= 0.80:
        return "stable", None
    if score >= 0.50:
        return "moderately_stable", reason
    if score >= 0.20:
        return "unstable", reason
    return "irregular_shape", reason or "pulse shapes have low correlation"


def _dominant_reason(
    shape_similarity: float,
    amplitude_variability: float,
    duration_variability: float,
) -> str | None:
    if shape_similarity < 0.75:
        return "pulse shapes have low correlation"
    if amplitude_variability > 0.30:
        return "high amplitude variability"
    if duration_variability > 0.25:
        return "high duration variability"
    return None


def _rounded_median(values: Sequence[float] | np.ndarray, ndigits: int = 3) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return _finite_round(float(np.median(array)), ndigits=ndigits)


def _finite_round(value: float, ndigits: int = 3) -> float | None:
    if not np.isfinite(value):
        return None
    return round(float(value), ndigits)


def _rounded_template(template: np.ndarray) -> list[float]:
    return [round(float(value), 4) for value in template if np.isfinite(value)]


def _empty_result(
    shape_quality: str,
    reason: str,
    *,
    shape_score: float | None = None,
) -> WaveformMorphology:
    return WaveformMorphology(
        enabled=True,
        valid_pulse_count=0,
        shape_quality=shape_quality,
        reason=reason,
        shape_score=shape_score,
    )


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))

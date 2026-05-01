from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from app.core.config import AppConfig
from app.core.ids import QUALITY_PREFIX, new_internal_id, new_public_id


FEATURE_NAMES = [
    "peak_count",
    "valid_peak_count",
    "ibi_cv",
    "perfusion_index",
    "template_corr_mean",
    "relative_power_hr_band",
    "baseline_drift",
    "flatline_ratio",
    "saturation_ratio",
    "outlier_ratio",
]


class QualityModelUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class MorphologyResult:
    valid_pulse_count: int
    shape_similarity: float | None
    average_pulse_template: list[float]
    amplitude_variability: float | None
    duration_variability: float | None
    morphology_variability: float | None
    shape_score: float | None
    shape_quality: str
    reason: str | None


@dataclass(frozen=True)
class QualityAnalysisResult:
    id: str
    public_id: str
    timestamp: datetime
    model: dict[str, object]
    quality_result: dict[str, object]
    features: dict[str, float | int | None]
    morphology: MorphologyResult


class QualityAnalyzer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: Any | None = None
        self._model_path: str | None = None

    def analyze(
        self,
        *,
        raw_ir: np.ndarray,
        raw_red: np.ndarray,
        filtered_ir: np.ndarray,
        fs: float,
    ) -> QualityAnalysisResult:
        model = self._load_model()
        timestamp = datetime.now(UTC)
        morphology = extract_morphology(filtered_ir=filtered_ir, fs=fs, config=self._config)
        features = extract_quality_features(
            raw_ir=raw_ir,
            raw_red=raw_red,
            filtered_ir=filtered_ir,
            fs=fs,
            config=self._config,
            morphology=morphology,
        )
        level, score = self._predict_quality(model, features)
        reason = None if level == "high" else _quality_reason(features)
        analysis_id = new_internal_id()
        return QualityAnalysisResult(
            id=analysis_id,
            public_id=new_public_id(QUALITY_PREFIX, analysis_id),
            timestamp=timestamp,
            model={
                "type": self._config.quality_model_type,
                "name": self._config.quality_model_name,
                "version": self._config.quality_model_version,
            },
            quality_result={
                "level": level,
                "score": round(float(score), 3),
                "reason": reason,
            },
            features=features,
            morphology=morphology,
        )

    def _load_model(self) -> Any:
        model_path = self._config.quality_model_path
        if not model_path:
            raise QualityModelUnavailable("QUALITY_MODEL_PATH is not configured")
        path = Path(model_path)
        if not path.exists():
            raise QualityModelUnavailable(f"quality model file does not exist: {model_path}")
        if self._model is not None and self._model_path == model_path:
            return self._model

        try:
            if path.suffix.lower() == ".json":
                self._model = _JsonThresholdModel(json.loads(path.read_text(encoding="utf-8")))
            elif path.suffix.lower() == ".joblib":
                try:
                    import joblib
                except ImportError as exc:
                    raise QualityModelUnavailable("joblib is required to load .joblib quality models") from exc
                self._model = joblib.load(path)
            else:
                with path.open("rb") as model_file:
                    self._model = pickle.load(model_file)
        except Exception as exc:
            raise QualityModelUnavailable(f"failed to load quality model: {exc}") from exc

        self._model_path = model_path
        return self._model

    @staticmethod
    def _predict_quality(model: Any, features: dict[str, float | int | None]) -> tuple[str, float]:
        vector = np.asarray([[float(features.get(name) or 0.0) for name in FEATURE_NAMES]], dtype=float)
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(vector), dtype=float)[0]
            classes = list(getattr(model, "classes_", ["low", "medium", "high"]))
            best_index = int(np.argmax(probabilities))
            return _normalize_label(classes[best_index]), float(probabilities[best_index])
        if hasattr(model, "predict"):
            prediction = model.predict(vector)
            label = prediction[0] if isinstance(prediction, (list, tuple, np.ndarray)) else prediction
            return _normalize_label(label), _score_for_label(label)
        raise QualityModelUnavailable("quality model must provide predict or predict_proba")


class _JsonThresholdModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def predict(self, vector: np.ndarray) -> list[str]:
        values = vector[0]
        features = dict(zip(FEATURE_NAMES, values, strict=True))
        high_threshold = float(self._payload.get("high_threshold", 0.75))
        medium_threshold = float(self._payload.get("medium_threshold", 0.45))
        score = (
            0.35 * features["template_corr_mean"]
            + 0.25 * features["relative_power_hr_band"]
            + 0.20 * max(0.0, 1.0 - features["ibi_cv"])
            + 0.20 * max(0.0, 1.0 - features["outlier_ratio"])
        )
        if score >= high_threshold:
            return ["high"]
        if score >= medium_threshold:
            return ["medium"]
        return ["low"]


def extract_quality_features(
    *,
    raw_ir: np.ndarray,
    raw_red: np.ndarray,
    filtered_ir: np.ndarray,
    fs: float,
    config: AppConfig,
    morphology: MorphologyResult | None = None,
) -> dict[str, float | int | None]:
    morphology = morphology or extract_morphology(filtered_ir=filtered_ir, fs=fs, config=config)
    peaks = _find_peaks(filtered_ir, fs, config)
    intervals = np.diff(peaks) / fs if peaks.size >= 2 else np.empty(0, dtype=float)
    valid_intervals = intervals[
        (intervals >= 60.0 / config.max_bpm)
        & (intervals <= 60.0 / config.min_bpm)
    ]
    ibi_cv = _coefficient_of_variation(valid_intervals)
    ir_dc = float(np.mean(raw_ir)) if raw_ir.size else 0.0
    ir_ac = float(np.sqrt(np.mean(np.square(filtered_ir)))) if filtered_ir.size else 0.0
    perfusion_index = (ir_ac / ir_dc) * 100.0 if ir_dc > 0 and ir_ac > 0 else None
    return {
        "peak_count": int(peaks.size),
        "valid_peak_count": int(morphology.valid_pulse_count),
        "ibi_cv": _rounded(ibi_cv),
        "perfusion_index": _rounded(perfusion_index, 4),
        "template_corr_mean": _rounded(morphology.shape_similarity),
        "relative_power_hr_band": _rounded(_relative_power_hr_band(filtered_ir, fs, config)),
        "baseline_drift": _rounded(_baseline_drift(raw_ir)),
        "flatline_ratio": _rounded(_flatline_ratio(raw_ir)),
        "saturation_ratio": _rounded(_saturation_ratio(raw_ir, raw_red, config)),
        "outlier_ratio": _rounded(_outlier_ratio(filtered_ir)),
    }


def extract_morphology(
    *,
    filtered_ir: np.ndarray,
    fs: float,
    config: AppConfig,
) -> MorphologyResult:
    if filtered_ir.size < max(3, int(fs)):
        return _empty_morphology("insufficient_pulses", "not enough samples")
    if not np.all(np.isfinite(filtered_ir)):
        return _empty_morphology("invalid_signal", "signal contains non-finite values")
    if float(np.ptp(filtered_ir)) <= 1e-9:
        return _empty_morphology("low_amplitude", "filtered IR channel is flat")

    peaks = _find_peaks(filtered_ir, fs, config)
    if peaks.size < 3:
        return _empty_morphology("insufficient_pulses", "not enough peaks")

    pulses: list[np.ndarray] = []
    amplitudes: list[float] = []
    durations: list[float] = []
    rise_times: list[float] = []
    widths_50: list[float] = []
    for index in range(1, peaks.size - 1):
        left_peak = int(peaks[index - 1])
        peak = int(peaks[index])
        right_peak = int(peaks[index + 1])
        left_foot = left_peak + int(np.argmin(filtered_ir[left_peak:peak + 1]))
        right_foot = peak + int(np.argmin(filtered_ir[peak:right_peak + 1]))
        if right_foot <= left_foot:
            continue
        duration = (right_foot - left_foot) / fs
        if duration < 60.0 / config.max_bpm or duration > 60.0 / config.min_bpm:
            continue
        pulse = filtered_ir[left_foot : right_foot + 1]
        amplitude = float(np.max(pulse) - np.min(pulse))
        if amplitude <= max(1e-9, 0.05 * float(np.std(filtered_ir))):
            continue
        normalized = (pulse - float(np.min(pulse))) / amplitude
        pulses.append(_resample(normalized, 100))
        amplitudes.append(amplitude)
        durations.append(duration)
        rise_times.append((peak - left_foot) / fs)
        widths_50.append(_pulse_width_50(normalized, fs))

    if len(pulses) < 3:
        return _empty_morphology("insufficient_pulses", "less than three valid pulse segments")

    keep = _non_outlier_mask(np.asarray(amplitudes, dtype=float))
    pulses = [pulse for pulse, keep_value in zip(pulses, keep, strict=True) if keep_value]
    amplitudes = [value for value, keep_value in zip(amplitudes, keep, strict=True) if keep_value]
    durations = [value for value, keep_value in zip(durations, keep, strict=True) if keep_value]
    rise_times = [value for value, keep_value in zip(rise_times, keep, strict=True) if keep_value]
    widths_50 = [value for value, keep_value in zip(widths_50, keep, strict=True) if keep_value]
    if len(pulses) < 3:
        return _empty_morphology("irregular_shape", "pulse amplitudes are inconsistent")

    pulse_matrix = np.vstack(pulses)
    template = np.mean(pulse_matrix, axis=0)
    correlations = [_correlation(pulse, template) for pulse in pulses]
    shape_similarity = float(np.median(correlations))
    amplitude_variability = _coefficient_of_variation(np.asarray(amplitudes, dtype=float))
    duration_variability = _coefficient_of_variation(np.asarray(durations, dtype=float))
    rise_variability = _coefficient_of_variation(np.asarray(rise_times, dtype=float)) or 0.0
    width_variability = _coefficient_of_variation(np.asarray(widths_50, dtype=float)) or 0.0
    morphology_variability = float(np.median(np.std(pulse_matrix, axis=0)))
    score = _clamp(
        0.45 * max(0.0, shape_similarity)
        + 0.20 * max(0.0, 1.0 - (amplitude_variability or 1.0))
        + 0.20 * max(0.0, 1.0 - (duration_variability or 1.0))
        + 0.10 * min(1.0, len(pulses) / 8.0)
        + 0.05 * max(0.0, 1.0 - (rise_variability + width_variability) / 2.0)
    )
    if score >= 0.8:
        quality = "stable"
    elif score >= 0.5:
        quality = "moderately_stable"
    elif score >= 0.2:
        quality = "unstable"
    else:
        quality = "irregular_shape"

    return MorphologyResult(
        valid_pulse_count=len(pulses),
        shape_similarity=round(shape_similarity, 3),
        average_pulse_template=[round(float(value), 4) for value in template.tolist()],
        amplitude_variability=_rounded(amplitude_variability),
        duration_variability=_rounded(duration_variability),
        morphology_variability=_rounded(morphology_variability),
        shape_score=round(score, 3),
        shape_quality=quality,
        reason=None if quality in {"stable", "moderately_stable"} else quality,
    )


def _find_peaks(filtered_ir: np.ndarray, fs: float, config: AppConfig) -> np.ndarray:
    signal_std = float(np.std(filtered_ir))
    if signal_std <= 1e-9:
        return np.empty(0, dtype=int)
    min_distance = max(1, int(round(fs * 60.0 / config.max_bpm)))
    prominence = max(signal_std * 0.35, 1e-9)
    peaks, _ = signal.find_peaks(filtered_ir, distance=min_distance, prominence=prominence)
    return peaks


def _relative_power_hr_band(filtered_ir: np.ndarray, fs: float, config: AppConfig) -> float | None:
    if filtered_ir.size < 4:
        return None
    frequencies, powers = signal.welch(filtered_ir, fs=fs, nperseg=min(filtered_ir.size, int(max(fs * 4, 8))))
    total = float(np.sum(powers))
    if total <= 0:
        return None
    band = (frequencies >= config.min_bpm / 60.0) & (frequencies <= config.max_bpm / 60.0)
    return float(np.sum(powers[band]) / total)


def _baseline_drift(raw_ir: np.ndarray) -> float | None:
    if raw_ir.size < 4:
        return None
    span = float(np.ptp(raw_ir))
    if span <= 0:
        return 0.0
    window = max(3, raw_ir.size // 10)
    kernel = np.ones(window) / window
    baseline = np.convolve(raw_ir, kernel, mode="valid")
    return float(np.ptp(baseline) / span)


def _flatline_ratio(raw_ir: np.ndarray) -> float:
    if raw_ir.size < 2:
        return 1.0
    threshold = max(1e-9, float(np.ptp(raw_ir)) * 0.001)
    return float(np.mean(np.abs(np.diff(raw_ir)) <= threshold))


def _saturation_ratio(raw_ir: np.ndarray, raw_red: np.ndarray, config: AppConfig) -> float:
    if raw_ir.size == 0 or raw_red.size == 0:
        return 0.0
    saturated = (raw_ir >= config.max_sensor_dc) | (raw_red >= config.max_sensor_dc)
    return float(np.mean(saturated))


def _outlier_ratio(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-9:
        return 0.0
    robust_z = np.abs(values - median) / (1.4826 * mad)
    return float(np.mean(robust_z > 5.0))


def _quality_reason(features: dict[str, float | int | None]) -> str:
    if (features.get("saturation_ratio") or 0.0) > 0.01:
        return "saturation"
    if (features.get("flatline_ratio") or 0.0) > 0.8:
        return "flatline"
    perfusion_index = features.get("perfusion_index")
    if perfusion_index is None or float(perfusion_index) < 0.05:
        return "low_perfusion"
    if (features.get("ibi_cv") or 0.0) > 0.25 or int(features.get("valid_peak_count") or 0) < 3:
        return "irregular_rhythm"
    return "high_noise"


def _empty_morphology(quality: str, reason: str) -> MorphologyResult:
    return MorphologyResult(
        valid_pulse_count=0,
        shape_similarity=None,
        average_pulse_template=[],
        amplitude_variability=None,
        duration_variability=None,
        morphology_variability=None,
        shape_score=None,
        shape_quality=quality,
        reason=reason,
    )


def _resample(values: np.ndarray, size: int) -> np.ndarray:
    source_x = np.linspace(0.0, 1.0, values.size)
    target_x = np.linspace(0.0, 1.0, size)
    return np.interp(target_x, source_x, values)


def _pulse_width_50(normalized: np.ndarray, fs: float) -> float:
    above = np.flatnonzero(normalized >= 0.5)
    if above.size < 2:
        return 0.0
    return float((above[-1] - above[0]) / fs)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) <= 1e-9 or float(np.std(right)) <= 1e-9:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _coefficient_of_variation(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    mean = float(np.mean(values))
    if mean <= 0:
        return None
    return float(np.std(values) / mean)


def _non_outlier_mask(values: np.ndarray) -> np.ndarray:
    if values.size < 4:
        return np.ones(values.size, dtype=bool)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return np.ones(values.size, dtype=bool)
    return (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)


def _normalize_label(label: object) -> str:
    if isinstance(label, (int, np.integer)):
        return {0: "low", 1: "medium", 2: "high"}.get(int(label), "low")
    normalized = str(label).lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    return "low"


def _score_for_label(label: object) -> float:
    return {"high": 0.9, "medium": 0.6, "low": 0.2}.get(_normalize_label(label), 0.2)


def _rounded(value: float | int | None, digits: int = 3) -> float | int | None:
    if value is None:
        return None
    if not np.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))

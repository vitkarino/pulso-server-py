from dataclasses import dataclass

import numpy as np
from scipy import signal

from app.config import AppConfig
from app.models import SignalQuality, WaveformMorphology
from app.processing.morphology import analyze_waveform_morphology


@dataclass(frozen=True)
class CalculationResult:
    bpm: float | None
    spo2: float | None
    ratio: float | None
    sensor_confidence: float
    quality: SignalQuality
    waveform_morphology: WaveformMorphology


@dataclass(frozen=True)
class BPMEstimate:
    bpm: float | None
    peak_count: int
    regularity_score: float
    reason: str | None = None


@dataclass(frozen=True)
class SpO2Estimate:
    spo2: float | None
    ratio: float | None
    perfusion_index: float | None
    reason: str | None = None


class VitalSignsCalculator:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def calculate(
        self,
        raw_ir: np.ndarray,
        raw_red: np.ndarray,
        filtered_ir: np.ndarray,
        filtered_red: np.ndarray,
        fs: float,
    ) -> CalculationResult:
        window_seconds = len(raw_ir) / fs
        if window_seconds < self._config.min_window_seconds:
            morphology = self._analyze_morphology(raw_ir, filtered_ir, np.empty(0, dtype=int), fs)
            confidence = self._clamp(window_seconds / self._config.min_window_seconds * 0.2)
            quality = SignalQuality(
                level="warming_up",
                samples_in_window=len(raw_ir),
                window_seconds=round(window_seconds, 2),
                shape_score=morphology.shape_score,
                reason="collecting enough samples for a stable window",
            )
            return CalculationResult(
                bpm=None,
                spo2=None,
                ratio=None,
                sensor_confidence=round(confidence, 3),
                quality=quality,
                waveform_morphology=morphology,
            )

        peaks = self._find_peaks(filtered_ir, fs)
        morphology = self._analyze_morphology(raw_ir, filtered_ir, peaks, fs)
        spo2_estimate = self._estimate_spo2(raw_ir, raw_red, filtered_ir, filtered_red, peaks)
        contact_problem = self._contact_problem(raw_ir, raw_red, spo2_estimate.perfusion_index)
        if contact_problem is not None:
            quality = SignalQuality(
                level="no_contact",
                samples_in_window=len(raw_ir),
                window_seconds=round(window_seconds, 2),
                perfusion_index=round(spo2_estimate.perfusion_index, 4)
                if spo2_estimate.perfusion_index is not None
                else None,
                peak_count=0,
                shape_score=morphology.shape_score,
                reason=contact_problem,
            )
            return CalculationResult(
                bpm=None,
                spo2=None,
                ratio=None,
                sensor_confidence=0.0,
                quality=quality,
                waveform_morphology=morphology,
            )

        bpm_estimate = self._estimate_bpm(filtered_ir, fs, peaks)
        confidence = self._sensor_confidence(
            window_seconds=window_seconds,
            bpm=bpm_estimate.bpm,
            spo2=spo2_estimate.spo2,
            ratio=spo2_estimate.ratio,
            peak_count=bpm_estimate.peak_count,
            regularity_score=bpm_estimate.regularity_score,
            perfusion_index=spo2_estimate.perfusion_index,
            shape_score=morphology.shape_score,
        )
        level = self._quality_level(
            bpm=bpm_estimate.bpm,
            spo2=spo2_estimate.spo2,
            peak_count=bpm_estimate.peak_count,
            perfusion_index=spo2_estimate.perfusion_index,
            confidence=confidence,
        )
        reason = self._combine_reasons(bpm_estimate.reason, spo2_estimate.reason)

        quality = SignalQuality(
            level=level,
            samples_in_window=len(raw_ir),
            window_seconds=round(window_seconds, 2),
            perfusion_index=round(spo2_estimate.perfusion_index, 4)
            if spo2_estimate.perfusion_index is not None
            else None,
            peak_count=bpm_estimate.peak_count,
            shape_score=morphology.shape_score,
            reason=reason,
        )
        return CalculationResult(
            bpm=bpm_estimate.bpm,
            spo2=spo2_estimate.spo2,
            ratio=spo2_estimate.ratio,
            sensor_confidence=round(confidence, 3),
            quality=quality,
            waveform_morphology=morphology,
        )

    def _analyze_morphology(
        self,
        raw_ir: np.ndarray,
        filtered_ir: np.ndarray,
        peaks: np.ndarray,
        fs: float,
    ) -> WaveformMorphology:
        return analyze_waveform_morphology(
            raw_ir=raw_ir,
            filtered_ir=filtered_ir,
            peaks=peaks,
            fs=fs,
            min_bpm=self._config.min_bpm,
            max_bpm=self._config.max_bpm,
        )

    def _find_peaks(self, filtered_ir: np.ndarray, fs: float) -> np.ndarray:
        signal_std = float(np.std(filtered_ir))
        if signal_std <= 1e-9:
            return np.empty(0, dtype=int)

        min_distance = max(1, int(round(fs * 60.0 / self._config.max_bpm)))
        prominence = max(signal_std * 0.35, 1e-9)
        peaks, _ = signal.find_peaks(
            filtered_ir,
            distance=min_distance,
            prominence=prominence,
        )
        if len(peaks) >= self._config.min_peaks:
            return peaks

        median = float(np.median(filtered_ir))
        mad = float(np.median(np.abs(filtered_ir - median)))
        if mad <= 0:
            return peaks
        relaxed = max(1.4826 * mad * 0.5, 1e-9)
        if relaxed >= prominence:
            return peaks
        retry, _ = signal.find_peaks(
            filtered_ir,
            distance=min_distance,
            prominence=relaxed,
        )
        return retry if len(retry) > len(peaks) else peaks

    def _estimate_bpm(self, filtered_ir: np.ndarray, fs: float, peaks: np.ndarray) -> BPMEstimate:
        signal_std = float(np.std(filtered_ir))
        if signal_std <= 1e-9:
            return BPMEstimate(bpm=None, peak_count=0, regularity_score=0.0, reason="filtered IR channel is flat")

        if len(peaks) >= self._config.min_peaks:
            intervals = np.diff(peaks) / fs
            valid = intervals[
                (intervals >= 60.0 / self._config.max_bpm)
                & (intervals <= 60.0 / self._config.min_bpm)
            ]
            valid = self._reject_interval_outliers(valid)
            if valid.size:
                bpm = 60.0 / float(np.median(valid))
                regularity_score = self._interval_regularity_score(valid)
                return BPMEstimate(
                    bpm=round(bpm, 1),
                    peak_count=int(len(peaks)),
                    regularity_score=regularity_score,
                )

        spectral_bpm = self._spectral_bpm(filtered_ir, fs)
        if spectral_bpm is not None:
            return BPMEstimate(
                bpm=spectral_bpm,
                peak_count=int(len(peaks)),
                regularity_score=0.45,
                reason="used spectral fallback because too few valid peaks",
            )

        return BPMEstimate(
            bpm=None,
            peak_count=int(len(peaks)),
            regularity_score=0.0,
            reason="not enough valid peaks",
        )

    def _spectral_bpm(self, filtered_ir: np.ndarray, fs: float) -> float | None:
        if filtered_ir.size < int(round(fs * self._config.min_window_seconds)):
            return None

        nperseg = min(filtered_ir.size, int(round(fs * 8)))
        frequencies, power = signal.welch(filtered_ir, fs=fs, nperseg=nperseg)
        mask = (
            (frequencies >= self._config.min_bpm / 60.0)
            & (frequencies <= self._config.max_bpm / 60.0)
        )
        if not np.any(mask):
            return None

        band_power = power[mask]
        band_freq = frequencies[mask]
        peak_index = int(np.argmax(band_power))
        peak_power = float(band_power[peak_index])
        noise_floor = float(np.median(band_power)) + 1e-12
        if peak_power / noise_floor < 3.0:
            return None

        chosen_freq = float(band_freq[peak_index])
        # Guard against locking onto the 2nd harmonic of the pulse: if a comparable
        # peak sits near f/2 inside the valid band, prefer the lower fundamental.
        half_freq = chosen_freq / 2.0
        if half_freq >= self._config.min_bpm / 60.0:
            tolerance = max(0.1 * half_freq, frequencies[1] - frequencies[0] if frequencies.size > 1 else 0.0)
            half_mask = (band_freq >= half_freq - tolerance) & (band_freq <= half_freq + tolerance)
            if np.any(half_mask):
                half_band = band_power[half_mask]
                half_power = float(np.max(half_band))
                if half_power >= peak_power * 0.5:
                    half_freqs = band_freq[half_mask]
                    chosen_freq = float(half_freqs[int(np.argmax(half_band))])
        return round(chosen_freq * 60.0, 1)

    @staticmethod
    def _reject_interval_outliers(intervals: np.ndarray) -> np.ndarray:
        if intervals.size < 4:
            return intervals
        q1, q3 = np.percentile(intervals, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            return intervals
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return intervals[(intervals >= lower) & (intervals <= upper)]

    @classmethod
    def _interval_regularity_score(cls, intervals: np.ndarray) -> float:
        if intervals.size < 2:
            return 0.4
        mean_interval = float(np.mean(intervals))
        if mean_interval <= 0:
            return 0.0
        cv = float(np.std(intervals) / mean_interval)
        return cls._clamp(1.0 - cv / 0.18)

    def _estimate_spo2(
        self,
        raw_ir: np.ndarray,
        raw_red: np.ndarray,
        filtered_ir: np.ndarray,
        filtered_red: np.ndarray,
        peaks: np.ndarray,
    ) -> SpO2Estimate:
        ir_dc = float(np.mean(raw_ir))
        red_dc = float(np.mean(raw_red))
        ir_ac = self._pulsatile_ac(filtered_ir, peaks)
        red_ac = self._pulsatile_ac(filtered_red, peaks)

        if min(ir_dc, red_dc) <= 0 or ir_ac is None or red_ac is None or min(ir_ac, red_ac) <= 0:
            return SpO2Estimate(spo2=None, ratio=None, perfusion_index=None, reason="SpO2 not reported: no usable AC/DC components")

        perfusion_index = (ir_ac / ir_dc) * 100.0
        ratio = (red_ac / red_dc) / (ir_ac / ir_dc)
        if not np.isfinite(ratio) or not 0.2 <= ratio <= 3.0:
            return SpO2Estimate(
                spo2=None,
                ratio=None,
                perfusion_index=float(perfusion_index),
                reason="SpO2 not reported: red/IR ratio is outside expected range",
            )

        if perfusion_index < self._config.min_spo2_perfusion_index:
            return SpO2Estimate(
                spo2=None,
                ratio=round(float(ratio), 4),
                perfusion_index=float(perfusion_index),
                reason="SpO2 not reported: perfusion index is too low for oxygen estimation",
            )

        spo2 = (
            self._config.spo2_a * ratio * ratio
            + self._config.spo2_b * ratio
            + self._config.spo2_c
            + self._config.spo2_offset
        )
        spo2 = max(0.0, min(100.0, spo2))
        return SpO2Estimate(
            spo2=round(float(spo2), 1),
            ratio=round(float(ratio), 4),
            perfusion_index=float(perfusion_index),
        )

    @staticmethod
    def _pulsatile_ac(filtered: np.ndarray, peaks: np.ndarray) -> float | None:
        if peaks.size >= 2:
            amplitudes: list[float] = []
            for i in range(peaks.size - 1):
                start = int(peaks[i])
                end = int(peaks[i + 1])
                if end <= start:
                    continue
                segment = filtered[start : end + 1]
                trough = float(np.min(segment))
                amplitude = float(filtered[start]) - trough
                if amplitude > 0:
                    amplitudes.append(amplitude)
            if amplitudes:
                return float(np.median(amplitudes))

        rms = float(np.sqrt(np.mean(np.square(filtered))))
        if rms <= 0:
            return None
        return rms * 2.0 * float(np.sqrt(2.0))

    def _contact_problem(
        self,
        raw_ir: np.ndarray,
        raw_red: np.ndarray,
        perfusion_index: float | None,
    ) -> str | None:
        ir_dc = float(np.mean(raw_ir))
        red_dc = float(np.mean(raw_red))

        if ir_dc < self._config.min_ir_dc or red_dc < self._config.min_red_dc:
            return "finger not detected: optical DC level is too low"
        if ir_dc > self._config.max_sensor_dc or red_dc > self._config.max_sensor_dc:
            return "finger not detected: sensor channel is saturated"
        if perfusion_index is None:
            return "finger not detected: no usable pulsatile component"
        if perfusion_index < self._config.min_perfusion_index:
            return "finger not detected: perfusion index is too low"
        if perfusion_index > self._config.max_perfusion_index:
            return "finger not detected: optical signal is unstable or exposed"
        return None

    @staticmethod
    def _quality_level(
        bpm: float | None,
        spo2: float | None,
        peak_count: int,
        perfusion_index: float | None,
        confidence: float,
    ) -> str:
        if bpm is None or spo2 is None:
            return "low"
        if confidence < 0.35 or perfusion_index is None or perfusion_index < 0.25 or peak_count < 4:
            return "low"
        if confidence < 0.7 or perfusion_index < 0.5 or peak_count < 6:
            return "medium"
        return "high"

    def _sensor_confidence(
        self,
        window_seconds: float,
        bpm: float | None,
        spo2: float | None,
        ratio: float | None,
        peak_count: int,
        regularity_score: float,
        perfusion_index: float | None,
        shape_score: float | None,
    ) -> float:
        window_score = self._clamp(window_seconds / self._config.min_window_seconds)
        pi_score = 0.0
        if perfusion_index is not None:
            pi_score = self._clamp((perfusion_index - 0.15) / 0.85)

        peak_score = self._clamp(peak_count / 8.0)
        ratio_score = self._ratio_confidence(ratio)
        confidence = (
            0.20 * window_score
            + 0.25 * pi_score
            + 0.30 * regularity_score
            + 0.15 * peak_score
            + 0.10 * ratio_score
        )

        if bpm is None:
            confidence *= 0.45
        if spo2 is None:
            confidence *= 0.70
        if shape_score is not None:
            confidence = confidence * 0.85 + shape_score * 0.15
        return self._clamp(confidence)

    @classmethod
    def _ratio_confidence(cls, ratio: float | None) -> float:
        if ratio is None:
            return 0.0
        if 0.4 <= ratio <= 1.3:
            return 1.0
        if 0.2 <= ratio < 0.4:
            return cls._clamp((ratio - 0.2) / 0.2)
        if 1.3 < ratio <= 3.0:
            return cls._clamp(1.0 - (ratio - 1.3) / 1.7)
        return 0.0

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _combine_reasons(*reasons: str | None) -> str | None:
        clean = [reason for reason in reasons if reason]
        if not clean:
            return None
        return "; ".join(clean)

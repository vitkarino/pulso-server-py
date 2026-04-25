import numpy as np
from scipy import signal

from app.config import AppConfig


class PPGNoiseFilter:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def filter_pair(self, ir: np.ndarray, red: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        return self.filter_channel(ir, fs), self.filter_channel(red, fs)

    def filter_channel(self, values: np.ndarray, fs: float) -> np.ndarray:
        if values.size < 3:
            return values.astype(float)

        cleaned = self._hampel(values)
        detrended = signal.detrend(cleaned, type="linear")

        nyquist = fs / 2.0
        high = min(self._config.bandpass_high_hz, nyquist * 0.95)
        low = min(self._config.bandpass_low_hz, high * 0.5)
        if low <= 0 or high <= low:
            return detrended

        sos = signal.butter(
            self._config.filter_order,
            [low, high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        try:
            return signal.sosfiltfilt(sos, detrended)
        except ValueError:
            return signal.sosfilt(sos, detrended)

    @staticmethod
    def _hampel(values: np.ndarray, window_size: int = 5, n_sigmas: float = 3.0) -> np.ndarray:
        series = np.asarray(values, dtype=float).copy()
        if series.size < (window_size * 2) + 1:
            return series

        result = series.copy()
        scale = 1.4826
        for index in range(window_size, series.size - window_size):
            window = series[index - window_size : index + window_size + 1]
            median = float(np.median(window))
            mad = float(scale * np.median(np.abs(window - median)))
            if mad > 0 and abs(series[index] - median) > n_sigmas * mad:
                result[index] = median
        return result

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

        centered = np.asarray(values, dtype=float) - float(np.mean(values))

        nyquist = fs / 2.0
        high = min(self._config.bandpass_high_hz, nyquist * 0.95)
        low = min(self._config.bandpass_low_hz, high * 0.5)
        if low <= 0 or high <= low:
            return centered

        sos = signal.butter(
            self._config.filter_order,
            [low, high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )

        try:
            return signal.sosfiltfilt(sos, centered)
        except ValueError:
            return signal.sosfilt(sos, centered)

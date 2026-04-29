from collections import deque
from dataclasses import dataclass
from threading import RLock

import numpy as np

from app.core.config import AppConfig
from app.schemas.device import PPGSample


@dataclass(frozen=True)
class SignalWindow:
    ir: np.ndarray
    red: np.ndarray
    fs: float
    samples_seen: int

    @property
    def seconds(self) -> float:
        return len(self.ir) / self.fs


class DeviceSignalBuffer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = RLock()
        self._ir: deque[float] = deque()
        self._red: deque[float] = deque()
        self._fs: float | None = None
        self._samples_seen = 0

    def append(self, samples: list[PPGSample], fs: float) -> SignalWindow:
        with self._lock:
            if self._fs != fs:
                self._ir.clear()
                self._red.clear()
                self._fs = fs
                self._samples_seen = 0

            max_samples = max(1, int(round(self._config.max_window_seconds * fs)))
            for sample in samples:
                self._ir.append(float(sample.ir))
                self._red.append(float(sample.red))
                self._samples_seen += 1

            while len(self._ir) > max_samples:
                self._ir.popleft()
                self._red.popleft()

            return SignalWindow(
                ir=np.asarray(self._ir, dtype=float),
                red=np.asarray(self._red, dtype=float),
                fs=fs,
                samples_seen=self._samples_seen,
            )

    def clear(self) -> None:
        with self._lock:
            self._ir.clear()
            self._red.clear()
            self._fs = None
            self._samples_seen = 0


class DeviceBufferRegistry:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._lock = RLock()
        self._buffers: dict[str, DeviceSignalBuffer] = {}

    def window_for(self, device_id: str, samples: list[PPGSample], fs: float) -> SignalWindow:
        with self._lock:
            buffer = self._buffers.get(device_id)
            if buffer is None:
                buffer = DeviceSignalBuffer(self._config)
                self._buffers[device_id] = buffer
        return buffer.append(samples=samples, fs=fs)

    def reset(self, device_id: str) -> None:
        with self._lock:
            buffer = self._buffers.get(device_id)
        if buffer is not None:
            buffer.clear()

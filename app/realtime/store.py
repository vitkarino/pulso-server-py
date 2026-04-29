from threading import RLock

from app.schemas.metrics import VitalSigns


class MetricsStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._latest: dict[str, VitalSigns] = {}

    def update(self, metrics: VitalSigns) -> None:
        with self._lock:
            self._latest[metrics.device_id] = metrics

    def get(self, device_id: str) -> VitalSigns | None:
        with self._lock:
            return self._latest.get(device_id)

    def delete(self, device_id: str) -> bool:
        with self._lock:
            return self._latest.pop(device_id, None) is not None

    def all(self) -> dict[str, VitalSigns]:
        with self._lock:
            return dict(self._latest)


metrics_store = MetricsStore()

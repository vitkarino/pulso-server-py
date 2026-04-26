from dataclasses import dataclass
import os


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    ws_port: int = _int_env("WS_PORT", 8080)
    database_url: str | None = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")

    min_bpm: float = _float_env("MIN_BPM", 40.0)
    max_bpm: float = _float_env("MAX_BPM", 180.0)
    bandpass_low_hz: float = _float_env("BANDPASS_LOW_HZ", 0.5)
    bandpass_high_hz: float = _float_env("BANDPASS_HIGH_HZ", 3.0)
    filter_order: int = _int_env("FILTER_ORDER", 4)

    min_window_seconds: float = _float_env("MIN_WINDOW_SECONDS", 8.0)
    max_window_seconds: float = _float_env("MAX_WINDOW_SECONDS", 20.0)
    measurement_duration_seconds: float = _float_env("MEASUREMENT_DURATION_SECONDS", 15.0)
    print_live_metrics: bool = _bool_env("PRINT_LIVE_METRICS", False)
    min_peaks: int = _int_env("MIN_PEAKS", 4)

    min_ir_dc: float = _float_env("MIN_IR_DC", 50_000.0)
    min_red_dc: float = _float_env("MIN_RED_DC", 10_000.0)
    max_sensor_dc: float = _float_env("MAX_SENSOR_DC", 250_000.0)
    min_perfusion_index: float = _float_env("MIN_PERFUSION_INDEX", 0.05)
    min_spo2_perfusion_index: float = _float_env("MIN_SPO2_PERFUSION_INDEX", 0.05)
    max_perfusion_index: float = _float_env("MAX_PERFUSION_INDEX", 20.0)

    spo2_a: float = _float_env("SPO2_A", -45.060)
    spo2_b: float = _float_env("SPO2_B", 30.354)
    spo2_c: float = _float_env("SPO2_C", 94.845)
    spo2_offset: float = _float_env("SPO2_OFFSET", 0.0)


settings = AppConfig()

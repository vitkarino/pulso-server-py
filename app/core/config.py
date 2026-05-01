from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import quote


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not name or name in os.environ:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


_load_dotenv()


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


def _database_url() -> str | None:
    explicit_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
    if explicit_url:
        return explicit_url

    component_names = {
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_PASSWORD",
        "POSTGRES_USER",
        "POSTGRES_DB",
        "PGHOST",
        "PGPORT",
        "PGPASSWORD",
        "PGUSER",
        "PGDATABASE",
    }
    if not _bool_env("DATABASE_ENABLED", False) and not any(os.getenv(name) for name in component_names):
        return None

    user = os.getenv("POSTGRES_USER") or os.getenv("PGUSER") or "vitkarino"
    database = os.getenv("POSTGRES_DB") or os.getenv("PGDATABASE") or "pulso"
    host = os.getenv("POSTGRES_HOST") or os.getenv("PGHOST") or "localhost"
    port = os.getenv("POSTGRES_PORT") or os.getenv("PGPORT") or "5432"
    password = os.getenv("POSTGRES_PASSWORD") or os.getenv("PGPASSWORD")

    auth = quote(user, safe="")
    if password:
        auth = f"{auth}:{quote(password, safe='')}"
    return f"postgresql+psycopg://{auth}@{host}:{port}/{quote(database, safe='')}"


@dataclass(frozen=True)
class AppConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    ws_port: int = _int_env("WS_PORT", 8080)
    database_url: str | None = _database_url()

    min_bpm: float = _float_env("MIN_BPM", 40.0)
    max_bpm: float = _float_env("MAX_BPM", 180.0)
    bandpass_low_hz: float = _float_env("BANDPASS_LOW_HZ", 0.5)
    bandpass_high_hz: float = _float_env("BANDPASS_HIGH_HZ", 3.0)
    filter_order: int = _int_env("FILTER_ORDER", 4)

    min_window_seconds: float = _float_env("MIN_WINDOW_SECONDS", 8.0)
    stable_spo2_window_seconds: float = _float_env("STABLE_SPO2_WINDOW_SECONDS", 10.0)
    spo2_warmup_cut_seconds: float = _float_env("SPO2_WARMUP_CUT_SECONDS", 2.0)
    max_window_seconds: float = _float_env("MAX_WINDOW_SECONDS", 20.0)
    measurement_duration_seconds: float = _float_env("MEASUREMENT_DURATION_SECONDS", 15.0)
    print_live_metrics: bool = _bool_env("PRINT_LIVE_METRICS", False)
    min_peaks: int = _int_env("MIN_PEAKS", 4)
    quality_model_path: str | None = os.getenv("QUALITY_MODEL_PATH")
    quality_model_type: str = os.getenv("QUALITY_MODEL_TYPE", "random_forest")
    quality_model_name: str = os.getenv("QUALITY_MODEL_NAME", "ppg_quality_rf")
    quality_model_version: str = os.getenv("QUALITY_MODEL_VERSION", "1.0.0")

    min_ir_dc: float = _float_env("MIN_IR_DC", 50_000.0)
    min_red_dc: float = _float_env("MIN_RED_DC", 10_000.0)
    max_sensor_dc: float = _float_env("MAX_SENSOR_DC", 250_000.0)
    min_perfusion_index: float = _float_env("MIN_PERFUSION_INDEX", 0.05)
    min_spo2_perfusion_index: float = _float_env("MIN_SPO2_PERFUSION_INDEX", 0.05)
    max_perfusion_index: float = _float_env("MAX_PERFUSION_INDEX", 20.0)

    spo2_a: float = _float_env("SPO2_A", -45.060)
    spo2_b: float = _float_env("SPO2_B", 30.354)
    spo2_c: float = _float_env("SPO2_C", 94.845)
    spo2_offset: float = _float_env("SPO2_OFFSET", -0.7)


settings = AppConfig()

from __future__ import annotations

from uuid import uuid4


DEVICE_PREFIX = "dev_"
MEASUREMENT_PREFIX = "mes_"
RECORDING_PREFIX = "rec_"
USER_PREFIX = "usr_"
PROJECT_PREFIX = "prj_"
QUALITY_PREFIX = "qlt_"


def new_internal_id() -> str:
    return str(uuid4())


def new_public_id(prefix: str, internal_id: str | None = None) -> str:
    return prefixed_id(prefix, internal_id or new_internal_id())


def prefixed_id(prefix: str, value: object | None) -> str | None:
    if value is None:
        return None
    raw = str(value)
    if raw.startswith(prefix):
        return raw
    return f"{prefix}{raw}"


def strip_prefix(value: object | None, prefix: str) -> str | None:
    if value is None:
        return None
    raw = str(value)
    if raw.startswith(prefix):
        return raw[len(prefix) :]
    return raw


def public_device_id(device_id: object | None) -> str | None:
    return prefixed_id(DEVICE_PREFIX, strip_prefix(device_id, DEVICE_PREFIX))


def internal_device_id(device_id: object | None) -> str | None:
    return strip_prefix(device_id, DEVICE_PREFIX)


def public_measurement_id(measurement_id: object | None) -> str | None:
    return prefixed_id(MEASUREMENT_PREFIX, strip_prefix(measurement_id, MEASUREMENT_PREFIX))


def public_recording_id(recording_id: object | None) -> str | None:
    return prefixed_id(RECORDING_PREFIX, strip_prefix(recording_id, RECORDING_PREFIX))


def public_user_id(user_id: object | None) -> str | None:
    return prefixed_id(USER_PREFIX, strip_prefix(user_id, USER_PREFIX))


def public_project_id(project_id: object | None) -> str | None:
    return prefixed_id(PROJECT_PREFIX, strip_prefix(project_id, PROJECT_PREFIX))


def public_quality_id(quality_id: object | None) -> str | None:
    return prefixed_id(QUALITY_PREFIX, strip_prefix(quality_id, QUALITY_PREFIX))


def numeric_id(value: object | None, prefix: str) -> int | None:
    raw = strip_prefix(value, prefix)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def is_simulated_device(device_id: object | None) -> bool:
    raw = internal_device_id(device_id)
    return bool(raw and raw.startswith("sim-"))

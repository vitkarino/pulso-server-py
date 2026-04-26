from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    desc,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine


JSON_TYPE = JSON().with_variant(JSONB, "postgresql")
SAMPLE_ID_TYPE = BigInteger().with_variant(Integer, "sqlite")

metadata = MetaData()

recordings_table = Table(
    "recordings",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("user_name", String(255), nullable=True),
    Column("user_id", String(255), nullable=True),
    Column("project_name", String(255), nullable=True),
    Column("project_id", String(255), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("duration_ms", BigInteger, nullable=True),
    Column("bpm", Float, nullable=True),
    Column("spo2", Float, nullable=True),
    Column("status", String(32), nullable=False),
    Column("signal_type", String(16), nullable=False),
    Column("sample_rate", Float, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("signal_quality", JSON_TYPE, nullable=True),
    Column("sensor_temp", Float, nullable=True),
    Column("device_id", String(255), nullable=True),
    Column("perfusion_index", Float, nullable=True),
    Column("ratio", Float, nullable=True),
    Column("sensor_confidence", Float, nullable=True),
    Column("peak_count", BigInteger, nullable=True),
)

recording_samples_table = Table(
    "recordings_samples",
    metadata,
    Column("id", SAMPLE_ID_TYPE, primary_key=True, autoincrement=True),
    Column(
        "recording_id",
        String(36),
        ForeignKey("recordings.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("sample_index", BigInteger, nullable=False),
    Column("raw_data", JSON_TYPE, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("recording_id", "sample_index", name="uq_recordings_samples_index"),
)

Index("ix_recordings_started_at", recordings_table.c.started_at)
Index("ix_recordings_status", recordings_table.c.status)
Index("ix_recordings_samples_recording_id", recording_samples_table.c.recording_id)


class RecordingRepository:
    def __init__(self, database_url: str | None) -> None:
        self._engine: Engine | None = None
        if database_url:
            self._engine = create_engine(
                _normalize_database_url(database_url),
                pool_pre_ping=True,
                future=True,
            )

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    def create_schema(self) -> None:
        metadata.create_all(self._require_engine())

    def create_recording(self, values: dict[str, Any]) -> None:
        with self._require_engine().begin() as connection:
            connection.execute(insert(recordings_table).values(**values))

    def update_recording(self, recording_id: str, values: dict[str, Any]) -> None:
        if not values:
            return
        with self._require_engine().begin() as connection:
            connection.execute(
                update(recordings_table)
                .where(recordings_table.c.id == recording_id)
                .values(**values)
            )

    def insert_samples(self, recording_id: str, samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        rows = [dict(sample, recording_id=recording_id) for sample in samples]
        with self._require_engine().begin() as connection:
            connection.execute(insert(recording_samples_table), rows)

    def get_recording(self, recording_id: str) -> dict[str, Any] | None:
        statement = select(recordings_table).where(recordings_table.c.id == recording_id)
        with self._require_engine().connect() as connection:
            row = connection.execute(statement).mappings().first()
        return dict(row) if row is not None else None

    def list_recordings(
        self,
        *,
        limit: int | None,
        offset: int,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        statement = select(recordings_table)
        if date_from is not None:
            statement = statement.where(recordings_table.c.started_at >= date_from)
        if date_to is not None:
            statement = statement.where(recordings_table.c.started_at <= date_to)
        statement = statement.order_by(desc(recordings_table.c.started_at)).offset(offset)
        if limit is not None:
            statement = statement.limit(limit)

        with self._require_engine().connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def list_recording_samples(
        self,
        recording_id: str,
        *,
        limit: int | None,
        offset: int,
    ) -> list[dict[str, Any]]:
        statement = (
            select(recording_samples_table)
            .where(recording_samples_table.c.recording_id == recording_id)
            .order_by(recording_samples_table.c.sample_index)
            .offset(offset)
        )
        if limit is not None:
            statement = statement.limit(limit)

        with self._require_engine().connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("DATABASE_URL is not configured")
        return self._engine


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url

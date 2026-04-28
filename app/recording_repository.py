from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    cast,
    create_engine,
    delete,
    desc,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine


JSON_TYPE = JSON().with_variant(JSONB, "postgresql")
SAMPLE_ID_TYPE = BigInteger().with_variant(Integer, "sqlite")

metadata = MetaData()

users_table = Table(
    "users",
    metadata,
    Column("id", Integer, Identity(), primary_key=True),
    Column("name", String(255), nullable=False),
    Column("age", Integer, nullable=True),
    Column("sex", String(32), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("recordings_qty", Integer, nullable=False, default=0, server_default="0"),
)

projects_table = Table(
    "projects",
    metadata,
    Column("id", Integer, Identity(), primary_key=True),
    Column("title", String(255), nullable=False),
    Column("description", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("recordings_qty", Integer, nullable=False, default=0, server_default="0"),
)

project_users_table = Table(
    "project_users",
    metadata,
    Column(
        "project_id",
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "user_id",
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("assigned_at", DateTime(timezone=True), nullable=False),
)

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
Index("ix_project_users_user_id", project_users_table.c.user_id)


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
        engine = self._require_engine()
        metadata.create_all(engine)
        with engine.begin() as connection:
            self._refresh_recording_counts(connection)

    def list_users(
        self,
        *,
        project_id: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        statement = select(users_table).order_by(desc(users_table.c.created_at))
        if project_id is not None:
            statement = (
                select(users_table, project_users_table.c.assigned_at)
                .select_from(
                    project_users_table.join(
                        users_table,
                        project_users_table.c.user_id == users_table.c.id,
                    )
                )
                .where(project_users_table.c.project_id == project_id)
                .order_by(desc(project_users_table.c.assigned_at))
            )
        statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        with self._require_engine().connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def get_user(self, user_id: int) -> dict[str, Any] | None:
        with self._require_engine().connect() as connection:
            return self._get_by_id(connection, users_table, user_id)

    def create_user(self, values: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC)
        row = {
            "name": values["name"],
            "age": values.get("age"),
            "sex": values.get("sex"),
            "created_at": now,
            "updated_at": now,
            "recordings_qty": 0,
        }
        with self._require_engine().begin() as connection:
            result = connection.execute(insert(users_table).values(**row))
            user_id = self._inserted_id(result.inserted_primary_key)
            created = self._get_by_id(connection, users_table, user_id)
        if created is None:
            raise RuntimeError("failed to create user")
        return created

    def update_user(self, user_id: int, values: dict[str, Any]) -> dict[str, Any] | None:
        update_values = {key: value for key, value in values.items() if key in {"name", "age", "sex"}}
        if update_values:
            update_values["updated_at"] = datetime.now(UTC)
        with self._require_engine().begin() as connection:
            if update_values:
                connection.execute(
                    update(users_table)
                    .where(users_table.c.id == user_id)
                    .values(**update_values)
                )
            return self._get_by_id(connection, users_table, user_id)

    def delete_user(self, user_id: int) -> bool:
        with self._require_engine().begin() as connection:
            result = connection.execute(delete(users_table).where(users_table.c.id == user_id))
        return result.rowcount > 0

    def list_projects(
        self,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        statement = select(projects_table).order_by(desc(projects_table.c.created_at)).offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        with self._require_engine().connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def get_project(self, project_id: int) -> dict[str, Any] | None:
        with self._require_engine().connect() as connection:
            return self._get_by_id(connection, projects_table, project_id)

    def create_project(self, values: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC)
        row = {
            "title": values["title"],
            "description": values.get("description"),
            "created_at": now,
            "updated_at": now,
            "recordings_qty": 0,
        }
        with self._require_engine().begin() as connection:
            result = connection.execute(insert(projects_table).values(**row))
            project_id = self._inserted_id(result.inserted_primary_key)
            created = self._get_by_id(connection, projects_table, project_id)
        if created is None:
            raise RuntimeError("failed to create project")
        return created

    def update_project(self, project_id: int, values: dict[str, Any]) -> dict[str, Any] | None:
        update_values = {key: value for key, value in values.items() if key in {"title", "description"}}
        if update_values:
            update_values["updated_at"] = datetime.now(UTC)
        with self._require_engine().begin() as connection:
            if update_values:
                connection.execute(
                    update(projects_table)
                    .where(projects_table.c.id == project_id)
                    .values(**update_values)
                )
            return self._get_by_id(connection, projects_table, project_id)

    def delete_project(self, project_id: int) -> bool:
        with self._require_engine().begin() as connection:
            result = connection.execute(delete(projects_table).where(projects_table.c.id == project_id))
        return result.rowcount > 0

    def list_project_users(self, project_id: int) -> list[dict[str, Any]]:
        statement = (
            select(users_table, project_users_table.c.assigned_at)
            .select_from(
                project_users_table.join(
                    users_table,
                    project_users_table.c.user_id == users_table.c.id,
                )
            )
            .where(project_users_table.c.project_id == project_id)
            .order_by(desc(project_users_table.c.assigned_at))
        )
        with self._require_engine().connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def add_project_user(self, project_id: int, user_id: int) -> dict[str, Any]:
        with self._require_engine().begin() as connection:
            existing = self._get_project_user(connection, project_id, user_id)
            if existing is not None:
                return existing

            assigned_at = datetime.now(UTC)
            connection.execute(
                insert(project_users_table).values(
                    project_id=project_id,
                    user_id=user_id,
                    assigned_at=assigned_at,
                )
            )
            created = self._get_project_user(connection, project_id, user_id)
        if created is None:
            raise RuntimeError("failed to assign project user")
        return created

    def delete_project_user(self, project_id: int, user_id: int) -> bool:
        with self._require_engine().begin() as connection:
            result = connection.execute(
                delete(project_users_table).where(
                    project_users_table.c.project_id == project_id,
                    project_users_table.c.user_id == user_id,
                )
            )
        return result.rowcount > 0

    def create_recording(self, values: dict[str, Any]) -> None:
        with self._require_engine().begin() as connection:
            connection.execute(insert(recordings_table).values(**values))
            self._increment_recording_counts(connection, values)

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
        device_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        statement = select(recordings_table)
        if date_from is not None:
            statement = statement.where(recordings_table.c.started_at >= date_from)
        if date_to is not None:
            statement = statement.where(recordings_table.c.started_at <= date_to)
        if device_id is not None:
            statement = statement.where(recordings_table.c.device_id == device_id)
        if user_id is not None:
            statement = statement.where(recordings_table.c.user_id == user_id)
        if project_id is not None:
            statement = statement.where(recordings_table.c.project_id == project_id)
        if status is not None:
            statement = statement.where(recordings_table.c.status == status)
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

    @staticmethod
    def _inserted_id(primary_key: tuple[Any, ...]) -> int:
        if not primary_key or primary_key[0] is None:
            raise RuntimeError("database did not return inserted primary key")
        return int(primary_key[0])

    @staticmethod
    def _get_by_id(connection: Connection, table: Table, row_id: int) -> dict[str, Any] | None:
        row = connection.execute(select(table).where(table.c.id == row_id)).mappings().first()
        return dict(row) if row is not None else None

    @staticmethod
    def _get_project_user(
        connection: Connection,
        project_id: int,
        user_id: int,
    ) -> dict[str, Any] | None:
        statement = select(project_users_table).where(
            project_users_table.c.project_id == project_id,
            project_users_table.c.user_id == user_id,
        )
        row = connection.execute(statement).mappings().first()
        return dict(row) if row is not None else None

    @staticmethod
    def _increment_recording_counts(connection: Connection, values: dict[str, Any]) -> None:
        user_id = _coerce_int(values.get("user_id"))
        if user_id is not None:
            connection.execute(
                update(users_table)
                .where(users_table.c.id == user_id)
                .values(recordings_qty=users_table.c.recordings_qty + 1)
            )

        project_id = _coerce_int(values.get("project_id"))
        if project_id is not None:
            connection.execute(
                update(projects_table)
                .where(projects_table.c.id == project_id)
                .values(recordings_qty=projects_table.c.recordings_qty + 1)
            )

    @staticmethod
    def _refresh_recording_counts(connection: Connection) -> None:
        user_count = (
            select(func.count(recordings_table.c.id))
            .where(recordings_table.c.user_id == cast(users_table.c.id, String))
            .scalar_subquery()
        )
        project_count = (
            select(func.count(recordings_table.c.id))
            .where(recordings_table.c.project_id == cast(projects_table.c.id, String))
            .scalar_subquery()
        )
        connection.execute(update(users_table).values(recordings_qty=user_count))
        connection.execute(update(projects_table).values(recordings_qty=project_count))


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
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
    create_engine,
    delete,
    desc,
    func,
    insert,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Connection, Engine

from app.core.ids import (
    PROJECT_PREFIX,
    RECORDING_PREFIX,
    USER_PREFIX,
    numeric_id,
    prefixed_id,
    public_project_id,
    public_recording_id,
    public_user_id,
    strip_prefix,
)


JSON_TYPE = JSON().with_variant(JSONB, "postgresql")
SAMPLE_ID_TYPE = BigInteger().with_variant(Integer, "sqlite")

metadata = MetaData()

users_table = Table(
    "users",
    metadata,
    Column("id", Integer, Identity(), primary_key=True),
    Column("public_id", String(64), nullable=True, unique=True),
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
    Column("public_id", String(64), nullable=True, unique=True),
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

measurements_table = Table(
    "measurements",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("public_id", String(64), nullable=True, unique=True),
    Column("user_id", String(64), nullable=True),
    Column("project_id", String(64), nullable=True),
    Column("device_id", String(255), nullable=True),
    Column("active_recording_id", String(36), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("duration_ms", BigInteger, nullable=True),
    Column("status", String(32), nullable=False),
    Column("signal_type", String(16), nullable=False, default="IR+R"),
    Column("sample_rate", Float, nullable=True),
    Column("sensor_temp", Float, nullable=True),
    Column("bpm", Float, nullable=True),
    Column("spo2", Float, nullable=True),
    Column("ratio", Float, nullable=True),
    Column("signal_quality", JSON_TYPE, nullable=True),
    Column("peak_count", BigInteger, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

recordings_table = Table(
    "recordings",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("public_id", String(64), nullable=True, unique=True),
    Column("measurement_id", String(36), ForeignKey("measurements.id", ondelete="SET NULL"), nullable=True),
    Column("quality_analysis_id", String(36), nullable=True),
    Column("use_for_ml_training", Boolean, nullable=False, default=False, server_default="false"),
    Column("user_name", String(255), nullable=True),
    Column("user_id", String(64), nullable=True),
    Column("project_name", String(255), nullable=True),
    Column("project_id", String(64), nullable=True),
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
    Column("peak_count", BigInteger, nullable=True),
    Column("sample_start_index", BigInteger, nullable=True),
    Column("sample_end_index", BigInteger, nullable=True),
    Column("samples_count", BigInteger, nullable=True),
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

quality_analyses_table = Table(
    "quality_analyses",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("public_id", String(64), nullable=True, unique=True),
    Column("recording_id", String(36), ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("model", JSON_TYPE, nullable=False),
    Column("quality_result", JSON_TYPE, nullable=False),
    Column("features", JSON_TYPE, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

Index("ix_measurements_public_id", measurements_table.c.public_id)
Index("ix_measurements_status", measurements_table.c.status)
Index("ix_measurements_device_id", measurements_table.c.device_id)
Index("ix_recordings_public_id", recordings_table.c.public_id)
Index("ix_recordings_started_at", recordings_table.c.started_at)
Index("ix_recordings_status", recordings_table.c.status)
Index("ix_recordings_measurement_id", recordings_table.c.measurement_id)
Index("ix_recordings_samples_recording_id", recording_samples_table.c.recording_id)
Index("ix_quality_analyses_public_id", quality_analyses_table.c.public_id)
Index("ix_quality_analyses_recording_id", quality_analyses_table.c.recording_id)
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

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()

    def create_schema(self) -> None:
        engine = self._require_engine()
        metadata.create_all(engine)
        with engine.begin() as connection:
            self._ensure_schema_columns(connection)
            self._backfill_public_ids(connection)
            self._backfill_legacy_measurements(connection)
            self._drop_removed_columns(connection)
            self._refresh_recording_counts(connection)
            self._cancel_running_rows(connection)

    def list_users(
        self,
        *,
        project_id: str | int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        numeric_project_id = numeric_id(project_id, PROJECT_PREFIX) if project_id is not None else None
        statement = select(users_table).order_by(desc(users_table.c.created_at))
        if numeric_project_id is not None:
            statement = (
                select(users_table, project_users_table.c.assigned_at)
                .select_from(
                    project_users_table.join(
                        users_table,
                        project_users_table.c.user_id == users_table.c.id,
                    )
                )
                .where(project_users_table.c.project_id == numeric_project_id)
                .order_by(desc(project_users_table.c.assigned_at))
            )
        statement = statement.offset(offset)
        if limit is not None:
            statement = statement.limit(limit)
        with self._require_engine().connect() as connection:
            rows = connection.execute(statement).mappings().all()
            output = [dict(row) for row in rows]
            for row in output:
                row["projects_count"] = self._count_user_projects(connection, int(row["id"]))
        return output

    def get_user(self, user_id: str | int) -> dict[str, Any] | None:
        with self._require_engine().connect() as connection:
            row = self._get_user(connection, user_id)
            if row is not None:
                row["projects_count"] = self._count_user_projects(connection, int(row["id"]))
            return row

    def create_user(self, values: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC)
        row = {
            "public_id": None,
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
            connection.execute(
                update(users_table)
                .where(users_table.c.id == user_id)
                .values(public_id=public_user_id(user_id))
            )
            created = self._get_user(connection, user_id)
        if created is None:
            raise RuntimeError("failed to create user")
        return created

    def update_user(self, user_id: str | int, values: dict[str, Any]) -> dict[str, Any] | None:
        update_values = {key: value for key, value in values.items() if key in {"name", "age", "sex"}}
        if update_values:
            update_values["updated_at"] = datetime.now(UTC)
        with self._require_engine().begin() as connection:
            existing = self._get_user(connection, user_id)
            if existing is None:
                return None
            if update_values:
                connection.execute(
                    update(users_table)
                    .where(users_table.c.id == existing["id"])
                    .values(**update_values)
                )
            return self._get_user(connection, existing["id"])

    def delete_user(self, user_id: str | int) -> bool:
        with self._require_engine().begin() as connection:
            existing = self._get_user(connection, user_id)
            if existing is None:
                return False
            connection.execute(delete(project_users_table).where(project_users_table.c.user_id == existing["id"]))
            result = connection.execute(delete(users_table).where(users_table.c.id == existing["id"]))
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
            output = [dict(row) for row in rows]
            for row in output:
                row["users_count"] = self._count_project_users(connection, int(row["id"]))
        return output

    def get_project(self, project_id: str | int) -> dict[str, Any] | None:
        with self._require_engine().connect() as connection:
            row = self._get_project(connection, project_id)
            if row is not None:
                row["users_count"] = self._count_project_users(connection, int(row["id"]))
            return row

    def create_project(self, values: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(UTC)
        row = {
            "public_id": None,
            "title": values["title"],
            "description": values.get("description"),
            "created_at": now,
            "updated_at": now,
            "recordings_qty": 0,
        }
        with self._require_engine().begin() as connection:
            result = connection.execute(insert(projects_table).values(**row))
            project_id = self._inserted_id(result.inserted_primary_key)
            connection.execute(
                update(projects_table)
                .where(projects_table.c.id == project_id)
                .values(public_id=public_project_id(project_id))
            )
            created = self._get_project(connection, project_id)
        if created is None:
            raise RuntimeError("failed to create project")
        return created

    def update_project(self, project_id: str | int, values: dict[str, Any]) -> dict[str, Any] | None:
        update_values = {key: value for key, value in values.items() if key in {"title", "description"}}
        if update_values:
            update_values["updated_at"] = datetime.now(UTC)
        with self._require_engine().begin() as connection:
            existing = self._get_project(connection, project_id)
            if existing is None:
                return None
            if update_values:
                connection.execute(
                    update(projects_table)
                    .where(projects_table.c.id == existing["id"])
                    .values(**update_values)
                )
            return self._get_project(connection, existing["id"])

    def delete_project(self, project_id: str | int) -> bool:
        with self._require_engine().begin() as connection:
            existing = self._get_project(connection, project_id)
            if existing is None:
                return False
            connection.execute(delete(project_users_table).where(project_users_table.c.project_id == existing["id"]))
            result = connection.execute(delete(projects_table).where(projects_table.c.id == existing["id"]))
        return result.rowcount > 0

    def list_project_users(self, project_id: str | int) -> list[dict[str, Any]]:
        numeric_project_id = numeric_id(project_id, PROJECT_PREFIX)
        if numeric_project_id is None:
            return []
        statement = (
            select(users_table, project_users_table.c.assigned_at)
            .select_from(
                project_users_table.join(
                    users_table,
                    project_users_table.c.user_id == users_table.c.id,
                )
            )
            .where(project_users_table.c.project_id == numeric_project_id)
            .order_by(desc(project_users_table.c.assigned_at))
        )
        with self._require_engine().connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def add_project_user(self, project_id: str | int, user_id: str | int) -> dict[str, Any]:
        numeric_project_id = numeric_id(project_id, PROJECT_PREFIX)
        numeric_user_id = numeric_id(user_id, USER_PREFIX)
        if numeric_project_id is None or numeric_user_id is None:
            raise ValueError("project_id and user_id must be valid")
        with self._require_engine().begin() as connection:
            existing = self._get_project_user(connection, numeric_project_id, numeric_user_id)
            if existing is not None:
                return existing

            assigned_at = datetime.now(UTC)
            connection.execute(
                insert(project_users_table).values(
                    project_id=numeric_project_id,
                    user_id=numeric_user_id,
                    assigned_at=assigned_at,
                )
            )
            created = self._get_project_user(connection, numeric_project_id, numeric_user_id)
        if created is None:
            raise RuntimeError("failed to assign project user")
        return created

    def delete_project_user(self, project_id: str | int, user_id: str | int) -> bool:
        numeric_project_id = numeric_id(project_id, PROJECT_PREFIX)
        numeric_user_id = numeric_id(user_id, USER_PREFIX)
        if numeric_project_id is None or numeric_user_id is None:
            return False
        with self._require_engine().begin() as connection:
            result = connection.execute(
                delete(project_users_table).where(
                    project_users_table.c.project_id == numeric_project_id,
                    project_users_table.c.user_id == numeric_user_id,
                )
            )
        return result.rowcount > 0

    def create_measurement(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._require_engine().begin() as connection:
            connection.execute(insert(measurements_table).values(**values))
            row = self._get_external_id_row(connection, measurements_table, values["id"])
        if row is None:
            raise RuntimeError("failed to create measurement")
        return row

    def update_measurement(self, measurement_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
        if not values:
            return self.get_measurement(measurement_id)
        with self._require_engine().begin() as connection:
            internal_id = self._resolve_external_id(connection, measurements_table, measurement_id)
            if internal_id is None:
                return None
            connection.execute(
                update(measurements_table)
                .where(measurements_table.c.id == internal_id)
                .values(**values)
            )
            return self._get_external_id_row(connection, measurements_table, internal_id)

    def get_measurement(self, measurement_id: str) -> dict[str, Any] | None:
        with self._require_engine().connect() as connection:
            return self._get_external_id_row(connection, measurements_table, measurement_id)

    def list_measurements(
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
        statement = select(measurements_table)
        if date_from is not None:
            statement = statement.where(measurements_table.c.started_at >= date_from)
        if date_to is not None:
            statement = statement.where(measurements_table.c.started_at <= date_to)
        if device_id is not None:
            statement = statement.where(measurements_table.c.device_id == strip_prefix(device_id, "dev_"))
        if user_id is not None:
            statement = statement.where(measurements_table.c.user_id == public_user_id(user_id))
        if project_id is not None:
            statement = statement.where(measurements_table.c.project_id == public_project_id(project_id))
        if status is not None:
            statement = statement.where(measurements_table.c.status == status)
        statement = statement.order_by(desc(measurements_table.c.started_at)).offset(offset)
        if limit is not None:
            statement = statement.limit(limit)

        with self._require_engine().connect() as connection:
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def create_recording(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._require_engine().begin() as connection:
            connection.execute(insert(recordings_table).values(**values))
            self._increment_recording_counts(connection, values)
            row = self._get_external_id_row(connection, recordings_table, values["id"])
        if row is None:
            raise RuntimeError("failed to create recording")
        return row

    def update_recording(self, recording_id: str, values: dict[str, Any]) -> dict[str, Any] | None:
        if not values:
            return self.get_recording(recording_id)
        with self._require_engine().begin() as connection:
            internal_id = self._resolve_external_id(connection, recordings_table, recording_id)
            if internal_id is None:
                return None
            connection.execute(
                update(recordings_table)
                .where(recordings_table.c.id == internal_id)
                .values(**values)
            )
            return self._get_external_id_row(connection, recordings_table, internal_id)

    def delete_recording(self, recording_id: str) -> bool:
        with self._require_engine().begin() as connection:
            internal_id = self._resolve_external_id(connection, recordings_table, recording_id)
            if internal_id is None:
                return False
            connection.execute(delete(quality_analyses_table).where(quality_analyses_table.c.recording_id == internal_id))
            connection.execute(
                delete(recording_samples_table).where(recording_samples_table.c.recording_id == internal_id)
            )
            result = connection.execute(delete(recordings_table).where(recordings_table.c.id == internal_id))
            if result.rowcount > 0:
                self._refresh_recording_counts(connection)
        return result.rowcount > 0

    def insert_samples(self, recording_id: str, samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        with self._require_engine().begin() as connection:
            internal_id = self._resolve_external_id(connection, recordings_table, recording_id)
            if internal_id is None:
                return
            rows = [dict(sample, recording_id=internal_id) for sample in samples]
            connection.execute(insert(recording_samples_table), rows)

    def get_recording(self, recording_id: str) -> dict[str, Any] | None:
        with self._require_engine().connect() as connection:
            return self._get_external_id_row(connection, recordings_table, recording_id)

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
            statement = statement.where(recordings_table.c.device_id == strip_prefix(device_id, "dev_"))
        if user_id is not None:
            statement = statement.where(recordings_table.c.user_id == public_user_id(user_id))
        if project_id is not None:
            statement = statement.where(recordings_table.c.project_id == public_project_id(project_id))
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
        with self._require_engine().connect() as connection:
            internal_id = self._resolve_external_id(connection, recordings_table, recording_id)
            if internal_id is None:
                return []
            statement = (
                select(recording_samples_table)
                .where(recording_samples_table.c.recording_id == internal_id)
                .order_by(recording_samples_table.c.sample_index)
                .offset(offset)
            )
            if limit is not None:
                statement = statement.limit(limit)
            rows = connection.execute(statement).mappings().all()
        return [dict(row) for row in rows]

    def upsert_quality_analysis(self, values: dict[str, Any]) -> dict[str, Any]:
        with self._require_engine().begin() as connection:
            recording_id = self._resolve_external_id(connection, recordings_table, values["recording_id"])
            if recording_id is None:
                raise ValueError("recording not found")
            existing = connection.execute(
                select(quality_analyses_table).where(quality_analyses_table.c.recording_id == recording_id)
            ).mappings().first()
            payload = dict(values, recording_id=recording_id)
            if existing is None:
                connection.execute(insert(quality_analyses_table).values(**payload))
                analysis_id = payload["id"]
            else:
                analysis_id = existing["id"]
                update_values = {
                    "timestamp": payload["timestamp"],
                    "model": payload["model"],
                    "quality_result": payload["quality_result"],
                    "features": payload["features"],
                    "updated_at": payload["updated_at"],
                }
                connection.execute(
                    update(quality_analyses_table)
                    .where(quality_analyses_table.c.id == analysis_id)
                    .values(**update_values)
                )
            connection.execute(
                update(recordings_table)
                .where(recordings_table.c.id == recording_id)
                .values(quality_analysis_id=analysis_id, updated_at=datetime.now(UTC))
            )
            row = self._get_external_id_row(connection, quality_analyses_table, analysis_id)
        if row is None:
            raise RuntimeError("failed to store quality analysis")
        return row

    def get_quality_analysis_for_recording(self, recording_id: str) -> dict[str, Any] | None:
        with self._require_engine().connect() as connection:
            internal_id = self._resolve_external_id(connection, recordings_table, recording_id)
            if internal_id is None:
                return None
            row = connection.execute(
                select(quality_analyses_table).where(quality_analyses_table.c.recording_id == internal_id)
            ).mappings().first()
        return dict(row) if row is not None else None

    def _require_engine(self) -> Engine:
        if self._engine is None:
            raise RuntimeError("DATABASE_URL is not configured")
        return self._engine

    @staticmethod
    def _ensure_schema_columns(connection: Connection) -> None:
        inspector = inspect(connection)
        table_names = set(inspector.get_table_names())
        if "users" in table_names:
            _ensure_columns(
                connection,
                "users",
                {
                    "public_id": "VARCHAR(64)",
                    "recordings_qty": "INTEGER NOT NULL DEFAULT 0",
                },
            )
        if "projects" in table_names:
            _ensure_columns(
                connection,
                "projects",
                {
                    "public_id": "VARCHAR(64)",
                    "description": "TEXT",
                    "recordings_qty": "INTEGER NOT NULL DEFAULT 0",
                },
            )
        if "recordings" in table_names:
            _ensure_columns(
                connection,
                "recordings",
                {
                    "public_id": "VARCHAR(64)",
                    "measurement_id": "VARCHAR(36)",
                    "quality_analysis_id": "VARCHAR(36)",
                    "use_for_ml_training": "BOOLEAN NOT NULL DEFAULT FALSE",
                    "sample_start_index": "BIGINT",
                    "sample_end_index": "BIGINT",
                    "samples_count": "BIGINT",
                },
            )

    @staticmethod
    def _backfill_public_ids(connection: Connection) -> None:
        for row in connection.execute(select(users_table.c.id, users_table.c.public_id)).mappings().all():
            if not row["public_id"]:
                connection.execute(
                    update(users_table)
                    .where(users_table.c.id == row["id"])
                    .values(public_id=public_user_id(row["id"]))
                )

        for row in connection.execute(select(projects_table.c.id, projects_table.c.public_id)).mappings().all():
            if not row["public_id"]:
                connection.execute(
                    update(projects_table)
                    .where(projects_table.c.id == row["id"])
                    .values(public_id=public_project_id(row["id"]))
                )

        rows = connection.execute(
            select(
                recordings_table.c.id,
                recordings_table.c.public_id,
                recordings_table.c.user_id,
                recordings_table.c.project_id,
            )
        ).mappings().all()
        for row in rows:
            values: dict[str, Any] = {}
            if not row["public_id"]:
                values["public_id"] = public_recording_id(row["id"])
            if row["user_id"] is not None:
                values["user_id"] = public_user_id(row["user_id"])
            if row["project_id"] is not None:
                values["project_id"] = public_project_id(row["project_id"])
            if values:
                connection.execute(update(recordings_table).where(recordings_table.c.id == row["id"]).values(**values))

    @staticmethod
    def _backfill_legacy_measurements(connection: Connection) -> None:
        rows = connection.execute(
            select(recordings_table).where(recordings_table.c.measurement_id.is_(None))
        ).mappings().all()
        for row in rows:
            measurement_id = row["id"]
            existing = connection.execute(
                select(measurements_table.c.id).where(measurements_table.c.id == measurement_id)
            ).first()
            if existing is None:
                connection.execute(
                    insert(measurements_table).values(
                        id=measurement_id,
                        public_id=prefixed_id("mes_", measurement_id),
                        user_id=row["user_id"],
                        project_id=row["project_id"],
                        device_id=row["device_id"],
                        active_recording_id=None,
                        started_at=row["started_at"],
                        finished_at=row["finished_at"],
                        duration_ms=row["duration_ms"],
                        status=row["status"],
                        signal_type=row["signal_type"],
                        sample_rate=row["sample_rate"],
                        sensor_temp=row["sensor_temp"],
                        bpm=row["bpm"],
                        spo2=row["spo2"],
                        ratio=row["ratio"],
                        signal_quality=row["signal_quality"],
                        peak_count=row["peak_count"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                )
            connection.execute(
                update(recordings_table)
                .where(recordings_table.c.id == row["id"])
                .values(measurement_id=measurement_id)
            )

    @staticmethod
    def _drop_removed_columns(connection: Connection) -> None:
        inspector = inspect(connection)
        table_names = set(inspector.get_table_names())
        legacy_column = "sensor_" + "confidence"
        preparer = connection.dialect.identifier_preparer
        for table_name in ("measurements", "recordings"):
            if table_name not in table_names:
                continue
            column_names = {column["name"] for column in inspector.get_columns(table_name)}
            if legacy_column not in column_names:
                continue
            connection.execute(
                text(
                    f"ALTER TABLE {preparer.quote(table_name)} "
                    f"DROP COLUMN {preparer.quote(legacy_column)}"
                )
            )

    @staticmethod
    def _cancel_running_rows(connection: Connection) -> None:
        now = datetime.now(UTC)
        for table in (measurements_table, recordings_table):
            rows = connection.execute(
                select(table.c.id, table.c.started_at).where(table.c.status == "running")
            ).mappings().all()
            for row in rows:
                started_at = row["started_at"]
                duration_ms = None
                if isinstance(started_at, datetime):
                    if started_at.tzinfo is None:
                        started_at = started_at.replace(tzinfo=UTC)
                    duration_ms = int(round(max(0.0, (now - started_at).total_seconds()) * 1000))
                connection.execute(
                    update(table)
                    .where(table.c.id == row["id"])
                    .values(
                        status="cancelled",
                        finished_at=now,
                        duration_ms=duration_ms,
                        updated_at=now,
                    )
                )

    @staticmethod
    def _inserted_id(primary_key: tuple[Any, ...]) -> int:
        if not primary_key or primary_key[0] is None:
            raise RuntimeError("database did not return inserted primary key")
        return int(primary_key[0])

    @staticmethod
    def _get_user(connection: Connection, user_id: str | int) -> dict[str, Any] | None:
        numeric = numeric_id(user_id, USER_PREFIX)
        statement = select(users_table).where(users_table.c.public_id == public_user_id(user_id))
        if numeric is not None:
            statement = select(users_table).where(
                (users_table.c.id == numeric) | (users_table.c.public_id == public_user_id(user_id))
            )
        row = connection.execute(statement).mappings().first()
        return dict(row) if row is not None else None

    @staticmethod
    def _get_project(connection: Connection, project_id: str | int) -> dict[str, Any] | None:
        numeric = numeric_id(project_id, PROJECT_PREFIX)
        statement = select(projects_table).where(projects_table.c.public_id == public_project_id(project_id))
        if numeric is not None:
            statement = select(projects_table).where(
                (projects_table.c.id == numeric) | (projects_table.c.public_id == public_project_id(project_id))
            )
        row = connection.execute(statement).mappings().first()
        return dict(row) if row is not None else None

    @staticmethod
    def _get_external_id_row(connection: Connection, table: Table, row_id: str) -> dict[str, Any] | None:
        internal_id = RecordingRepository._resolve_external_id(connection, table, row_id)
        if internal_id is None:
            return None
        row = connection.execute(select(table).where(table.c.id == internal_id)).mappings().first()
        return dict(row) if row is not None else None

    @staticmethod
    def _resolve_external_id(connection: Connection, table: Table, row_id: str) -> str | None:
        row = connection.execute(
            select(table.c.id).where((table.c.id == row_id) | (table.c.public_id == row_id))
        ).first()
        return str(row[0]) if row is not None else None

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
    def _count_user_projects(connection: Connection, user_id: int) -> int:
        return int(
            connection.execute(
                select(func.count()).select_from(project_users_table).where(project_users_table.c.user_id == user_id)
            ).scalar_one()
        )

    @staticmethod
    def _count_project_users(connection: Connection, project_id: int) -> int:
        return int(
            connection.execute(
                select(func.count())
                .select_from(project_users_table)
                .where(project_users_table.c.project_id == project_id)
            ).scalar_one()
        )

    @staticmethod
    def _increment_recording_counts(connection: Connection, values: dict[str, Any]) -> None:
        user_public_id = public_user_id(values.get("user_id"))
        if user_public_id is not None:
            connection.execute(
                update(users_table)
                .where(users_table.c.public_id == user_public_id)
                .values(recordings_qty=users_table.c.recordings_qty + 1)
            )

        project_public_id = public_project_id(values.get("project_id"))
        if project_public_id is not None:
            connection.execute(
                update(projects_table)
                .where(projects_table.c.public_id == project_public_id)
                .values(recordings_qty=projects_table.c.recordings_qty + 1)
            )

    @staticmethod
    def _refresh_recording_counts(connection: Connection) -> None:
        for row in connection.execute(select(users_table.c.id, users_table.c.public_id)).mappings().all():
            count = connection.execute(
                select(func.count(recordings_table.c.id)).where(recordings_table.c.user_id == row["public_id"])
            ).scalar_one()
            connection.execute(
                update(users_table)
                .where(users_table.c.id == row["id"])
                .values(recordings_qty=int(count))
            )

        for row in connection.execute(select(projects_table.c.id, projects_table.c.public_id)).mappings().all():
            count = connection.execute(
                select(func.count(recordings_table.c.id)).where(recordings_table.c.project_id == row["public_id"])
            ).scalar_one()
            connection.execute(
                update(projects_table)
                .where(projects_table.c.id == row["id"])
                .values(recordings_qty=int(count))
            )


def _ensure_columns(connection: Connection, table_name: str, columns: dict[str, str]) -> None:
    existing = {column["name"] for column in inspect(connection).get_columns(table_name)}
    for column_name, ddl_type in columns.items():
        if column_name not in existing:
            connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl_type}"))


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg://", 1)
    return database_url

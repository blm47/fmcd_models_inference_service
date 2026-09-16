"""
Настройки подключений S3 и PostgreSQL читаются из ENV, настройки сервиса — только из YAML.
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.models.contracts import ModelSpec
from app.models.registry import validate_backend


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"Не задана переменная окружения {name}")
    return value


def env_bool(name: str) -> bool:
    value = required_env(name).lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"{name}: ожидается true, false, 1 или 0")
    return value in ("true", "1")


def positive_integer(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: ожидается положительное целое число")


@dataclass(frozen=True)
class S3Config:
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket_in: str
    bucket_out: str
    region: str
    use_ssl: bool
    verify_ssl: bool


@dataclass(frozen=True)
class TaskStoreConfig:
    backend: str
    state_key: str
    cas_timeout_sec: float
    cas_retry_interval_sec: float
    poll_interval_sec: float
    retention_months: int
    max_pending_tasks: int
    heartbeat_interval_sec: float
    task_timeout_sec: float
    cleanup_interval_sec: float
    progress_update_interval_sec: float
    connect_timeout_sec: float
    read_timeout_sec: float
    strip_etag_quotes: bool

    def __post_init__(self):
        for name, value in vars(self).items():
            if name.endswith("_sec"):
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"task_store.{name}: ожидается конечное положительное число")
        for name in ("retention_months", "max_pending_tasks"):
            positive_integer(f"task_store.{name}", getattr(self, name))
        if self.backend not in ("s3", "pg"):
            raise ValueError("task_store.backend: ожидается s3 или pg")
        if self.task_timeout_sec <= self.heartbeat_interval_sec:
            raise ValueError("task_timeout_sec должен быть больше heartbeat_interval_sec")
        if self.task_timeout_sec <= self.progress_update_interval_sec:
            raise ValueError("task_timeout_sec должен быть больше progress_update_interval_sec")
        if not isinstance(self.state_key, str) or not self.state_key.strip("/"):
            raise ValueError("task_store.state_key: ожидается непустой ключ S3")
        if self.state_key.startswith("/") or "://" in self.state_key:
            raise ValueError("task_store.state_key задаётся относительно выходного бакета")
        if type(self.strip_etag_quotes) is not bool:
            raise ValueError("strip_etag_quotes: ожидается YAML boolean")


@dataclass(frozen=True)
class PostgresConfig:
    url: str
    login: str
    password: str = field(repr=False)
    schema: str
    table_name: str
    sslmode: str = "prefer"

    def __post_init__(self):
        from urllib.parse import urlsplit

        if self.sslmode not in (
            "disable",
            "allow",
            "prefer",
            "require",
            "verify-ca",
            "verify-full",
        ):
            raise ValueError("POSTGRES_SSLMODE: неизвестный режим TLS")
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in ("postgresql", "postgres")
            or not parsed.hostname
            or not parsed.path.strip("/")
        ):
            raise ValueError("POSTGRES_URL: ожидается postgresql://host:port/database")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("POSTGRES_URL: credentials и query передаются отдельно")
        for name in ("schema", "table_name"):
            value = getattr(self, name)
            if not value or "\x00" in value or len(value.encode("utf-8")) > 63:
                raise ValueError(f"POSTGRES_{name.upper()}: некорректный идентификатор")


@dataclass(frozen=True)
class Settings:
    models: list[ModelSpec]
    s3: S3Config
    task_store: TaskStoreConfig
    postgres: PostgresConfig | None = None


def load_settings(config_path: str | Path = "configs/models.yaml") -> Settings:
    """
    Вызывается один раз в lifespan; ENV не переопределяет поля YAML.
    """
    with open(config_path, encoding="utf-8") as source:
        raw = yaml.safe_load(source)

    if not isinstance(raw, dict) or set(raw) != {"models", "task_store"}:
        raise ValueError("YAML должен содержать только models и task_store")
    if not isinstance(raw["models"], list):
        raise ValueError("models: ожидается список настроек моделей")
    models = []
    for model in raw["models"]:
        if not isinstance(model, dict):
            raise ValueError("models: ожидается словарь настроек каждой модели")
        spec = ModelSpec(**model)
        validate_backend(spec.backend)
        models.append(spec)

    if not models or len({model.name for model in models}) != len(models):
        raise ValueError("Список моделей должен быть непустым, имена — уникальными")

    task_store = TaskStoreConfig(**raw["task_store"])
    postgres = None
    if task_store.backend == "pg":
        postgres = PostgresConfig(
            url=required_env("POSTGRES_URL"),
            login=required_env("POSTGRES_LOGIN"),
            password=required_env("POSTGRES_PASSWORD"),
            schema=required_env("POSTGRES_SCHEMA"),
            table_name=required_env("POSTGRES_TABLE_NAME"),
            sslmode=os.environ.get("POSTGRES_SSLMODE", "prefer"),
        )
    return Settings(
        postgres=postgres,
        models=models,
        task_store=task_store,
        s3=S3Config(
            endpoint_url=required_env("S3_ENDPOINT_URL"),
            access_key=required_env("S3_ACCESS_KEY"),
            secret_key=required_env("S3_SECRET_KEY"),
            bucket_in=required_env("S3_BUCKET_IN"),
            bucket_out=required_env("S3_BUCKET_OUT"),
            region=required_env("S3_REGION"),
            use_ssl=env_bool("S3_USE_SSL"),
            verify_ssl=env_bool("S3_VERIFY_SSL"),
        ),
    )

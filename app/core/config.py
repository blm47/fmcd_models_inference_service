"""
Настройки S3 читаются из ENV, настройки сервиса — только из YAML.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


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
class ModelConfig:
    name: str
    weights_path: str
    schema_path: str
    calibrators_path: str
    freq_encoding_path: str
    id_cols: list[str]


@dataclass(frozen=True)
class InferenceConfig:
    device: str
    infer_batch_size: int
    chunk_size: int

    def __post_init__(self):
        for name in ("infer_batch_size", "chunk_size"):
            positive_integer(name, getattr(self, name))


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
    state_key: str
    cas_timeout_sec: float
    cas_retry_interval_sec: float
    poll_interval_sec: float
    retention_months: int
    max_pending_tasks: int
    heartbeat_interval_sec: float
    heartbeat_timeout_sec: float
    progress_timeout_sec: float
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
        if self.heartbeat_timeout_sec <= self.heartbeat_interval_sec:
            raise ValueError("heartbeat_timeout_sec должен быть больше heartbeat_interval_sec")
        if self.progress_timeout_sec <= self.progress_update_interval_sec:
            raise ValueError("progress_timeout_sec должен быть больше progress_update_interval_sec")
        if not isinstance(self.state_key, str) or not self.state_key.strip("/"):
            raise ValueError("task_store.state_key: ожидается непустой ключ S3")
        if self.state_key.startswith("/") or "://" in self.state_key:
            raise ValueError("task_store.state_key задаётся относительно выходного бакета")
        if type(self.strip_etag_quotes) is not bool:
            raise ValueError("strip_etag_quotes: ожидается YAML boolean")


@dataclass(frozen=True)
class Settings:
    models: list[ModelConfig]
    inference: InferenceConfig
    s3: S3Config
    task_store: TaskStoreConfig


def load_settings(config_path: str | Path = "configs/models.yaml") -> Settings:
    """
    Вызывается один раз в lifespan; ENV не переопределяет поля YAML.
    """
    with open(config_path, encoding="utf-8") as source:
        raw = yaml.safe_load(source)

    if set(raw) != {"models", "inference", "task_store"}:
        raise ValueError("YAML должен содержать только models, inference и task_store")
    
    models = []
    for model in raw["models"]:
        directory = Path(model["artifacts_dir"])
        models.append(
            ModelConfig(
                name=model["name"],
                weights_path=str(directory / model["weights_file"]),
                schema_path=str(directory / model["schema_file"]),
                calibrators_path=str(directory / model["calibrators_file"]),
                freq_encoding_path=str(directory / model["freq_encoding_file"]),
                id_cols=model["id_cols"],
            )
        )
        if (
            not isinstance(model["id_cols"], list)
            or not model["id_cols"]
            or not all(isinstance(col, str) and col for col in model["id_cols"])
        ):
            raise ValueError("id_cols должен содержать имена колонок идентификаторов")
        
    if not models or len({model.name for model in models}) != len(models):
        raise ValueError("Список моделей должен быть непустым, имена — уникальными")
    
    return Settings(
        models=models,
        inference=InferenceConfig(**raw["inference"]),
        task_store=TaskStoreConfig(**raw["task_store"]),
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

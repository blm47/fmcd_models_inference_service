"""
Централизованная конфигурация сервиса.

Паттерн чтения S3-настроек — по аналогии с вашим существующим сервисом
(BGEM3): в configs/models.yaml хранится не само значение секрета/эндпоинта,
а ИМЯ переменной окружения, откуда его нужно прочитать. Сами значения
приходят в контейнер как обычные переменные окружения — их туда
прокидывает Helm (из ConfigMap для несекретных и из Secret для секретных
значений, см. deploy/helm/templates/deployment.yaml).

Почему так, а не "${VAR}"-интерполяция прямо в yaml:
  - Явное разделение: в yaml видно, ЧТО является env-зависимым полем,
    а в env vars — реальные значения, которые различаются по кластерам
    (георезервирование - два кластера с разными S3-эндпоинтами).
  - Fail-fast: если обязательная переменная не выставлена, сервис падает
    при старте с понятной ошибкой, а не с невнятным KeyError в середине
    инференса на 2-3 млн строк.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Environment variable '{name}' is required")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


@dataclass
class ModelConfig:
    """Описание одной модели из списка models.yaml -> models[]."""

    name: str
    weights_path: str
    schema_path: str
    calibrators_path: str
    freq_encoding_path: str


@dataclass
class InferenceConfig:
    device: str = "cuda"
    infer_batch_size: int = 16
    chunk_size: int = 100_000


@dataclass
class S3Config:
    """Значения читаются из env по именам, указанным в models.yaml -> s3.*_env."""

    endpoint_url: str
    access_key: str
    secret_key: str
    bucket_in: str
    bucket_out: str
    region: str
    use_ssl: bool
    verify_ssl: bool


@dataclass
class TaskConfig:
    progress_update_interval_sec: int = 10


@dataclass
class TaskStoreConfig:
    """
    Конфигурация физического хранения состояния задач (task tracker).

    backend: "s3" (сейчас) | "redis" | "postgres" (заложено на будущее -
    смена значения не требует изменений в TaskManager/TaskStore, только
    в фабрике create_task_storage_backend()).

    retention_months читается из переменной окружения
    TASK_STORE_RETENTION_MONTHS, по умолчанию 6.
    """

    backend: str = "s3"
    state_key: str
    lock_key: str
    lease_seconds: int
    wait_timeout_sec: int
    poll_interval_sec: float
    retention_months: int


@dataclass
class Settings:
    models: list[ModelConfig]
    inference: InferenceConfig
    s3: S3Config
    task: TaskConfig
    task_store: TaskStoreConfig

    def get_model_config(self, name: str) -> ModelConfig:
        for m in self.models:
            if m.name == name:
                return m
        available = [m.name for m in self.models]
        raise KeyError(f"Модель '{name}' не описана в конфиге. Доступные: {available}")


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_settings_cache: Settings | None = None


def load_settings(config_path: str = "configs/models.yaml") -> Settings:
    """
    Синглтон настроек на весь жизненный цикл процесса. В отличие от pydantic-
    settings BaseSettings, здесь конфиг — простые dataclasses, т.к. основная
    валидация (обязательность env-переменных) уже делается в _env().
    """
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache

    raw = _load_yaml(config_path)

    models = [
        ModelConfig(
            name=m["name"],
            weights_path=str(Path(m["artifacts_dir"]) / m["weights_file"]),
            schema_path=str(Path(m["artifacts_dir"]) / m["schema_file"]),
            calibrators_path=str(Path(m["artifacts_dir"]) / m["calibrators_file"]),
            freq_encoding_path=str(Path(m["artifacts_dir"]) / m["freq_encoding_file"]),
        )
        for m in raw["models"]
    ]

    s3_raw = raw["s3"]
    s3 = S3Config(
        endpoint_url=_env(s3_raw["endpoint_url_env"]),
        access_key=_env(s3_raw["access_key_env"]),
        secret_key=_env(s3_raw["secret_key_env"]),
        bucket_in=_env(s3_raw["bucket_in_env"]),
        bucket_out=_env(s3_raw["bucket_out_env"]),
        region=_env(s3_raw["region_env"], "us-east-1"),
        use_ssl=_env_bool(s3_raw["use_ssl_env"], True),
        verify_ssl=_env_bool(s3_raw["verify_ssl_env"], True),
    )

    task_store_raw = raw.get("task_store", {})
    task_store = TaskStoreConfig(
        backend=os.environ.get("TASK_STORE_BACKEND", task_store_raw.get("backend", "s3")),
        state_key=task_store_raw.get("state_key"),
        lock_key=task_store_raw.get("lock_key"),
        lease_seconds=_env_int("TASK_STORE_LEASE_SECONDS", task_store_raw.get("lease_seconds")),
        wait_timeout_sec=_env_int("TASK_STORE_WAIT_TIMEOUT_SEC", task_store_raw.get("wait_timeout_sec")),
        poll_interval_sec=float(
            os.environ.get("TASK_STORE_POLL_INTERVAL_SEC", task_store_raw.get("poll_interval_sec"))
        ),
        retention_months=_env_int(
            "TASK_STORE_RETENTION_MONTHS", task_store_raw.get("retention_months", 6)
        ),
    )

    _settings_cache = Settings(
        models=models,
        inference=InferenceConfig(**raw.get("inference", {})),
        s3=s3,
        task=TaskConfig(**raw.get("task", {})),
        task_store=task_store,
    )
    return _settings_cache


def get_settings() -> Settings:
    """Удобный алиас для Depends(get_settings) в FastAPI-роутах."""
    return load_settings()

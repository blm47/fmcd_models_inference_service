"""
Единая точка выбора физического backend'а хранения состояния задач.

Смена движка (S3 -> Redis -> Postgres в будущем) = добавление ветки
здесь + новой реализации TaskStorageBackend. Ни TaskManager, ни
TaskStore, ни роуты не меняются.
"""

from __future__ import annotations

import s3fs

from app.core.config import S3Config, TaskStoreConfig
from app.tasks.backends.base import TaskStorageBackend
from app.tasks.backends.s3_backend import S3LeaseLockBackend


def create_task_storage_backend(
    task_store_config: TaskStoreConfig, s3_config: S3Config
) -> TaskStorageBackend:
    if task_store_config.backend == "s3":
        client_kwargs = {}
        if not s3_config.verify_ssl:
            client_kwargs["verify"] = False

        fs = s3fs.S3FileSystem(
            key=s3_config.access_key,
            secret=s3_config.secret_key,
            client_kwargs={
                "endpoint_url": s3_config.endpoint_url,
                "region_name": s3_config.region,
                **client_kwargs,
            },
            use_ssl=s3_config.use_ssl,
            skip_instance_cache=True,
            use_listings_cache=False,
        )
        bucket = s3_config.bucket_out.rstrip("/")
        state_key = f"{bucket}/{task_store_config.state_key.lstrip('/')}"
        lock_key = f"{bucket}/{task_store_config.lock_key.lstrip('/')}"

        return S3LeaseLockBackend(
            s3_fs=fs,
            state_key=state_key,
            lock_key=lock_key,
            lease_seconds=task_store_config.lease_seconds,
            wait_timeout_sec=task_store_config.wait_timeout_sec,
            poll_interval_sec=task_store_config.poll_interval_sec,
            retention_months=task_store_config.retention_months,
        )

    # Задел на будущее: 
    elif task_store_config.backend == "_redis": 
        pass
    elif task_store_config.backend == "_postgres":
        pass

    raise ValueError(
        f"Неизвестный backend task tracker'а: '{task_store_config.backend}'. "
        f"Поддерживается: 's3' (сейчас); 'redis'/'postgres' - заложено на будущее."
    )

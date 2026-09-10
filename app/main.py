"""
Точка входа FastAPI-приложения.

lifespan выполняет всю "тяжёлую" инициализацию один раз при старте пода:
  - загрузка конфига (env + config/models.yaml)
  - загрузка всех моделей из списка models[] на GPU + препроцессинг-
    артефактов в dict[str, ModelBundle]
  - создание TaskStore/CancellationRegistry/TaskManager/S3Client

Всё складывается в app.state, роуты достают через Depends (app/deps.py).
Ни один тяжёлый объект не создаётся на каждый запрос.
"""

from contextlib import asynccontextmanager
import os
import socket

from fastapi import FastAPI

from app.api.routes_infer import router as infer_router
from app.api.routes_tasks import router as tasks_router
from app.core.config import load_settings
from app.core.logging import setup_logging
from app.tasks.backends.factory import create_task_storage_backend
from app.models.loader import load_all_models
from app.storage.s3_client import S3Client
from app.tasks.cancellation import CancellationRegistry
from app.tasks.manager import TaskManager
from app.tasks.maintenance import TaskMaintenance
from app.tasks.state import TaskStore


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger = setup_logging()

    try:
        settings = load_settings()

        models = load_all_models(settings.models, settings.inference, logger)

        task_storage_backend = create_task_storage_backend(settings.task_store, settings.s3)
        task_storage_backend.initialize()

        pod_id = os.environ.get("POD_NAME") or os.environ.get("HOSTNAME", socket.gethostname())

        app.state.settings = settings
        app.state.models = models
        app.state.task_store = TaskStore(
            task_storage_backend,
            heartbeat_interval_sec=settings.task_store.heartbeat_interval_sec,
            heartbeat_timeout_sec=settings.task_store.heartbeat_timeout_sec,
        )
        app.state.cancellation_registry = CancellationRegistry()
        app.state.task_manager = TaskManager(
            app.state.task_store, app.state.cancellation_registry, pod_id=pod_id
        )
        app.state.s3_client = S3Client(settings.s3)
        app.state.pod_id = pod_id

        # Другой процесс не должен помечать старого исполнителя как FAILED только
        # по имени пода. API исключает его из активных после истечения heartbeat.
        maintenance = TaskMaintenance(app.state.task_store, settings.task_store.cleanup_interval_sec)
        maintenance.start()

        logger.info(f"Сервис запущен на поде pod_id={pod_id}")

    except Exception:
        logger.exception("Service initialization failed")
        raise

    try:
        yield
    finally:
        maintenance.stop()
    # На shutdown специально ничего не чистим: если под убивают во время
    # активной задачи, это внештатная ситуация уровня K8s (readiness/liveness),
    # а не штатный сценарий graceful shutdown в v1.


app = FastAPI(title="FMCD Inference Service", lifespan=lifespan)
app.include_router(infer_router)
app.include_router(tasks_router)
# app = FastAPI(title="FMCD Inference Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get('/healthz/readiness')
async def route_readiness_probe():
    """Проверка готовности сервиса (readiness probe)."""
    return {'details': 'OK'}


@app.get('/healthz/liveness')
async def route_liveness_probe():
    """Проверка доступности сервиса (liveness probe)."""
    return {'details': 'OK'}

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
import logging
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
from app.tasks.state import TaskStore


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger = setup_logging()

    try:
        settings = load_settings()

        models = load_all_models(settings.models, settings.inference, logger)

        task_storage_backend = create_task_storage_backend(settings.task_store, settings.s3)

        pod_id = os.environ.get("HOSTNAME", socket.gethostname())

        app.state.settings = settings
        app.state.models = models
        app.state.task_store = TaskStore(task_storage_backend)
        app.state.cancellation_registry = CancellationRegistry()
        app.state.task_manager = TaskManager(
            app.state.task_store, app.state.cancellation_registry, pod_id=pod_id
        )
        app.state.s3_client = S3Client(settings.s3)
        app.state.pod_id = pod_id

        # Убиваем повисшие таски в случае рестарта ПОДа
        active_tasks = app.state.task_store.get_all_active()
        local_active_task = [task.task_id for task in active_tasks if task.pod_id == pod_id]
        for bad_task_id in local_active_task:
            logger.warn(f"Found active task {bad_task_id} on pod {pod_id}, aborting it")
            app.state.task_store.set_status(bad_task_id, TaskStatus.FAILED)
            
        logger.info(f"Сервис запущен на поде pod_id={pod_id}")

    except Exception as exc:
        import traceback
        msg = f"lifespan error: {traceback.format_exc()}"
        print(msg, flush=True)
        logger.error(msg)

    yield
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
    """ readiness probe """
    return {'details': 'OK'}


@app.get('/healthz/liveness')
async def route_liveness_probe():
    """ liveness probe """
    return {'details': 'OK'}

"""
Точка входа FastAPI-приложения.

lifespan выполняет всю "тяжёлую" инициализацию один раз при старте пода:
  - загрузка конфига (env + configs/models.yaml)
  - загрузка всех моделей из списка models[] на GPU (H100) + препроцессинг-
    артефактов в dict[str, ModelBundle]
  - создание TaskStore/CancellationRegistry/TaskManager/S3Client

Всё складывается в app.state, роуты достают через Depends (app/deps.py).
Ни один тяжёлый объект не создаётся на каждый запрос.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes_infer import router as infer_router
from app.api.routes_tasks import router as tasks_router
from app.core.config import load_settings
from app.core.logging import setup_logging
from app.models.loader import load_all_models
from app.storage.s3_client import S3Client
from app.tasks.cancellation import CancellationRegistry
from app.tasks.manager import TaskManager
from app.tasks.state import TaskStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = load_settings()

    models = load_all_models(settings.models, settings.inference)

    app.state.settings = settings
    app.state.models = models
    app.state.task_store = TaskStore()
    app.state.cancellation_registry = CancellationRegistry()
    app.state.task_manager = TaskManager(app.state.task_store, app.state.cancellation_registry)
    app.state.s3_client = S3Client(settings.s3)

    yield
    # На shutdown специально ничего не чистим: если под убивают во время
    # активной задачи, это внештатная ситуация уровня K8s (readiness/liveness),
    # а не штатный сценарий graceful shutdown в v1.


app = FastAPI(title="FMCD Inference Service", lifespan=lifespan)
app.include_router(infer_router)
app.include_router(tasks_router)


@app.get("/health")
def health():
    return {"status": "ok"}

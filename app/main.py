"""
Инициализация настроек, моделей и единственного потока очереди на поде.
"""

import asyncio
import os
import socket
import threading
import traceback
from contextlib import asynccontextmanager

import urllib3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_infer import router as infer_router
from app.api.routes_tasks import router as tasks_router
from app.api.schemas import HealthResponse
from app.core.config import load_settings
from app.core.logging import setup_logging
from app.core.shutdown import install_shutdown_handlers, restore_shutdown_handlers
from app.models.artifacts import download_model_artifacts
from app.tasks.backends.base import TaskStorageError
from app.tasks.backends.factory import create_task_backend
from app.tasks.state import QueueConflictError, TaskStore
from app.tasks.worker import consume_queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = setup_logging()
    app.state.logger = logger
    store = None
    stop = threading.Event()
    shutdown_handlers = {}
    started_threads = []
    try:
        settings = load_settings()
        for model in settings.models:
            await asyncio.to_thread(
                download_model_artifacts, model.name, model.artifacts_dir, logger
            )
        s3 = settings.s3
        queue = settings.task_store
        if not s3.verify_ssl:
            # Убираем повторяющиеся предупреждения urllib3 при опросе очереди S3.
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warn("Проверка TLS-сертификата S3 отключена: S3_VERIFY_SSL=false")

        store = TaskStore(create_task_backend(settings, logger), queue, logger)
        await asyncio.to_thread(store.initialize)
        from app.storage.s3_client import S3Client

        models = {spec.name: spec for spec in settings.models}
        s3_client = S3Client(s3, logger)
        pod_id = os.environ.get("POD_NAME") or socket.gethostname()
        consumer = threading.Thread(
            target=consume_queue,
            args=(store, models, settings, s3_client, pod_id, stop, logger),
            name="inference-queue",
            daemon=True,
        )
        app.state.settings = settings
        app.state.models = models
        app.state.task_store = store
        app.state.s3_client = s3_client
        app.state.stop = stop
        app.state.consumer = consumer
        shutdown_handlers = install_shutdown_handlers(stop, logger)
        consumer.start()
        started_threads.append(consumer)
        logger.info(f"Сервис запущен на поде {pod_id}")
    except Exception:
        stop.set()
        for thread in started_threads:
            await asyncio.to_thread(thread.join)
        restore_shutdown_handlers(shutdown_handlers)
        if store is not None:
            store.close()
        logger.error(("Не удалось инициализировать сервис") + "\n" + traceback.format_exc())
        raise

    try:
        yield
    finally:
        stop.set()
        # Ждём текущую операцию перед закрытием S3; K8s ограничивает срок shutdown.
        await asyncio.to_thread(consumer.join)
        restore_shutdown_handlers(shutdown_handlers)
        store.close()
        logger.info(f"Сервис остановлен на поде {pod_id}")


app = FastAPI(
    title="FMCD Inference Service",
    description=(
        "Асинхронный GPU-инференс над parquet в S3. Любой под принимает заявки, "
        "свободный worker забирает их из общей очереди (S3 или PostgreSQL).\n\n"
        "**Airflow:** hadoop_to_S3_Operator запускает Spark job → `POST /infer` → опрашивать "
        "`GET /tasks/{task_id}/status` → при `DONE` S3_to_Hadoop_Operator запускает Spark job. "
        "Parquet передаётся между хранилищами через Spark.\n\n"
        "**Статусы:** `QUEUED → RUNNING → FINALIZING → DONE`. "
        "Ошибка или таймаут переводит задачу в `FAILED`. "
        "Отмена: `QUEUED → ABORTED` или `RUNNING → ABORTING → ABORTED`.\n\n"
        "Повтор HTTP-запроса использует прежний ключ идемпотентности. "
        "Новая попытка расчёта требует нового ключа и очищенного выходного префикса. "
    ),
    openapi_tags=[
        {"name": "inference", "description": "Постановка расчёта в общую очередь."},
        {"name": "tasks", "description": "Статусы, прогресс и отмена задач на всех подах."},
        {"name": "health", "description": "Проверки готовности процесса для Kubernetes."},
    ],
    swagger_ui_parameters={"defaultModelsExpandDepth": 0, "displayRequestDuration": True},
    lifespan=lifespan,
)
app.include_router(infer_router)
app.include_router(tasks_router)


@app.exception_handler(QueueConflictError)
async def queue_conflict(request: Request, exc: QueueConflictError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


async def storage_unavailable(request: Request, exc: Exception):
    request.app.state.logger.error(
        (f"Запрос к очереди не завершён: {exc}") + "\n" + "".join(traceback.format_exception(exc))
    )
    return JSONResponse(
        status_code=503,
        content={"detail": "Очередь недоступна; повторите запрос с тем же ключом идемпотентности"},
    )


for error_type in (ClientError, BotoCoreError, TimeoutError, TaskStorageError):
    app.add_exception_handler(error_type, storage_unavailable)


def _locally_alive(request: Request) -> bool:
    consumer = getattr(request.app.state, "consumer", None)
    stop = getattr(request.app.state, "stop", None)
    return consumer is not None and consumer.is_alive() and stop is not None and not stop.is_set()


def _health_response(available: bool):
    return JSONResponse(
        status_code=200 if available else 503,
        content={"status": "ok" if available else "unavailable"},
    )


@app.get(
    "/healthz/liveness",
    tags=["health"],
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "unavailable"}},
    summary="Проверить работу процесса и consumer",
)
async def liveness(request: Request):
    """
    Проверяет локальный consumer и сигнал остановки без обращения к хранилищу.
    """
    return _health_response(_locally_alive(request))


@app.get(
    "/health",
    tags=["health"],
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "unavailable"}},
    summary="Проверить готовность сервиса и очередь",
)
@app.get(
    "/healthz/readiness",
    tags=["health"],
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "unavailable"}},
    summary="Проверить готовность пода и таймауты задач",
)
def readiness(request: Request):
    """
    Проверяет consumer, доступность очереди и переводит просроченные задачи в FAILED.

    Запускает retention S3 по cleanup_interval_sec; PG не очищается.
    Этот вызов заменяет поток monitor_queue: вне Kubernetes ручку нужно опрашивать.
    При сбое S3 хранилища или уже выполняющейся проверке возвращает 503.
    Проверка выполняется и на поде с занятым GPU, heartbeat работает независимо.
    """
    if not _locally_alive(request):
        return _health_response(False)
    try:
        available = request.app.state.task_store.check_queue()
    except Exception:
        request.app.state.logger.error(
            "Не удалось проверить очередь в readiness\n" + traceback.format_exc()
        )
        return _health_response(False)
    return _health_response(available)

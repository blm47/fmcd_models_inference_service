"""Инициализация настроек, моделей и единственного потока очереди на поде."""

import asyncio
import os
import socket
import threading
import traceback
from contextlib import asynccontextmanager

import boto3
import urllib3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_infer import router as infer_router
from app.api.routes_tasks import router as tasks_router
from app.api.schemas import HealthResponse
from app.core.config import load_settings
from app.core.logging import setup_logging
from app.core.shutdown import install_shutdown_handlers, restore_shutdown_handlers
from app.tasks.state import QueueConflictError, TaskStore
from app.tasks.worker import consume_queue, monitor_queue


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = setup_logging()
    app.state.logger = logger
    client = None
    stop = threading.Event()
    previous_handlers = {}
    started_threads = []
    try:
        settings = load_settings()
        s3 = settings.s3
        queue = settings.task_store
        if not s3.verify_ssl:
            # Убираем повторяющиеся предупреждения urllib3 при опросе очереди S3.
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            logger.warn("Проверка TLS-сертификата S3 отключена: S3_VERIFY_SSL=false")
        client = boto3.client(
            "s3",
            endpoint_url=s3.endpoint_url,
            aws_access_key_id=s3.access_key,
            aws_secret_access_key=s3.secret_key,
            region_name=s3.region,
            use_ssl=s3.use_ssl,
            verify=s3.verify_ssl,
            config=Config(
                signature_version="s3v4",
                connect_timeout=queue.connect_timeout_sec,
                read_timeout=queue.read_timeout_sec,
                # Повторы CAS управляются TaskStore, а не скрытым retry SDK.
                retries={"mode": "standard", "total_max_attempts": 1},
            ),
        )
        store = TaskStore(client, s3.bucket_out, queue, logger)
        await asyncio.to_thread(store.initialize)
        from app.models.loader import load_all_models
        from app.storage.s3_client import S3Client

        models = await asyncio.to_thread(
            load_all_models, settings.models, settings.inference, logger
        )
        s3_client = S3Client(s3, logger)
        pod_id = os.environ.get("POD_NAME") or socket.gethostname()
        consumer = threading.Thread(
            target=consume_queue,
            args=(store, models, settings, s3_client, pod_id, stop, logger),
            name="inference-queue",
            daemon=True,
        )
        monitor = threading.Thread(
            target=monitor_queue, args=(store, stop, logger), name="queue-monitor", daemon=True
        )
        app.state.settings = settings
        app.state.models = models
        app.state.task_store = store
        app.state.s3_client = s3_client
        app.state.stop = stop
        app.state.consumer = consumer
        app.state.monitor = monitor
        previous_handlers = install_shutdown_handlers(stop, logger)
        monitor.start()
        started_threads.append(monitor)
        consumer.start()
        started_threads.append(consumer)
        logger.info(f"Сервис запущен на поде {pod_id}")
    except Exception:
        stop.set()
        for thread in started_threads:
            await asyncio.to_thread(thread.join)
        restore_shutdown_handlers(previous_handlers)
        if client is not None:
            client.close()
        logger.error(("Не удалось инициализировать сервис") + "\n" + traceback.format_exc())
        raise

    try:
        yield
    finally:
        stop.set()
        # Ждём текущую операцию перед закрытием S3; K8s ограничивает срок shutdown.
        await asyncio.to_thread(consumer.join)
        await asyncio.to_thread(monitor.join)
        restore_shutdown_handlers(previous_handlers)
        client.close()
        logger.info(f"Сервис остановлен на поде {pod_id}")


app = FastAPI(
    title="FMCD Inference Service",
    description=(
        "Асинхронный GPU-инференс над parquet в S3. Любой под принимает заявки, "
        "свободный worker забирает их из общей очереди S3.\n\n"
        "**Airflow:** подготовить вход → `POST /infer` → опрашивать "
        "`GET /tasks/{task_id}/status` → при `DONE` забрать результат в Hadoop.\n\n"
        "**Статусы:** `QUEUED → RUNNING → FINALIZING → DONE`. "
        "Ошибка или таймаут переводит задачу в `FAILED`. "
        "Отмена: `QUEUED → ABORTED` или `RUNNING → ABORTING → ABORTED`.\n\n"
        "Повтор HTTP-запроса использует прежний ключ идемпотентности. "
        "Новая попытка расчёта требует нового ключа и очищенного выходного префикса. "
        "Не удаляйте системный файл очереди. `FAILED` по таймауту не подтверждает "
        "остановку старого PUT: повтор в тот же физический путь не имеет строгой изоляции."
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
        (f"Запрос к очереди S3 не завершён: {exc}")
        + "\n"
        + "".join(traceback.format_exception(exc))
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Очередь S3 недоступна; повторите запрос с тем же ключом идемпотентности"
        },
    )


for error_type in (ClientError, BotoCoreError, TimeoutError):
    app.add_exception_handler(error_type, storage_unavailable)


@app.get(
    "/health",
    tags=["health"],
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "unavailable"}},
    summary="Проверить состояние сервиса",
)
@app.get(
    "/healthz/readiness",
    tags=["health"],
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "unavailable"}},
    summary="Проверить готовность пода",
)
@app.get(
    "/healthz/liveness",
    tags=["health"],
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "unavailable"}},
    summary="Проверить работу потоков пода",
)
def health(request: Request):
    """200 при работающих consumer и monitor, иначе 503. Доступ к S3 не проверяет."""
    consumer = getattr(request.app.state, "consumer", None)
    monitor = getattr(request.app.state, "monitor", None)
    stop = getattr(request.app.state, "stop", None)
    if (
        consumer is None
        or not consumer.is_alive()
        or monitor is None
        or not monitor.is_alive()
        or stop.is_set()
    ):
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok"}

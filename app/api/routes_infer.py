"""
POST /infer — единственный универсальный эндпоинт инференса (с полем model_name).

Синхронная часть (до ответа клиенту):
  1. Найти модель в dict моделей по model_name (404, если нет).
  2. Провалидировать наличие всех фич модели
     (404, если сам префикс не существует в S3; 422, если фич не хватает).
  3. Попытаться атомарно занять "слот" активной задачи (409, если уже занято).
Всё, что после — уходит в BackgroundTasks (сам инференс).
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.api.schemas import (
    InferRequest,
    S3PathNotFoundResponse,
    TaskAcceptedResponse,
    TaskBusyResponse,
    ValidationErrorResponse,
)
from app.core.logging import setup_logging
from app.core.config import Settings
from app.deps import (
    get_cancellation_registry,
    get_models,
    get_s3_client,
    get_settings,
    get_task_manager,
    get_task_store,
)
from app.models.registry import ModelBundle, get_model_bundle
from app.models.validation import validate_input_parquet
from app.storage.s3_client import S3Client
from app.tasks.cancellation import CancellationRegistry
from app.tasks.manager import TaskAlreadyRunningError, TaskManager
from app.tasks.state import TaskStore
from app.tasks.worker import run_task

# logger = logging.getLogger(__name__)
router = APIRouter(tags=["inference"])


@router.post(
    "/infer",
    response_model=TaskAcceptedResponse,
    responses={
        404: {"model": S3PathNotFoundResponse},
        409: {"model": TaskBusyResponse},
        422: {"model": ValidationErrorResponse},
    },
    status_code=202,
)
def infer(
    request: InferRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
    models: dict[str, ModelBundle] = Depends(get_models),
    task_manager: TaskManager = Depends(get_task_manager),
    task_store: TaskStore = Depends(get_task_store),
    cancellation: CancellationRegistry = Depends(get_cancellation_registry),
    s3_client: S3Client = Depends(get_s3_client),
    logger: logging.Logger = Depends(setup_logging),
):
    try:
        bundle = get_model_bundle(models, request.model_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        validation = validate_input_parquet(request.s3_input_path, bundle, s3_client)
    except FileNotFoundError as exc:
        # pyarrow.dataset кидает FileNotFoundError, если под s3_input_path
        # нет ни одного part-*.parquet файла
        logger.warning("Входной префикс не найден в S3: %s", request.s3_input_path)
        raise HTTPException(
            status_code=404,
            detail=S3PathNotFoundResponse(
                s3_path=request.s3_input_path,
                detail=str(exc),
            ).dict(),
        )

    if not validation.is_valid:
        raise HTTPException(
            status_code=422,
            detail=ValidationErrorResponse(missing_columns=validation.missing_columns).dict(),
        )

    if s3_client.prefix_has_parquet(request.s3_output_path):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "output_prefix_not_empty",
                "s3_output_path": request.s3_output_path,
                "detail": f"Префикс {request.s3_output_path} уже существует в S3. "
                    f"Сервис не перезаписывает существующие данные.",
            }
        )

    try:
        task = task_manager.try_start_task(
            model_name=request.model_name,
            s3_input_path=request.s3_input_path,
            s3_output_path=request.s3_output_path,
            total_rows=validation.total_rows,
        )
    except TaskAlreadyRunningError as exc:
        active = exc.active_task
        raise HTTPException(
            status_code=409,
            detail=TaskBusyResponse(
                task_id=active.task_id,
                status=active.status,
                progress_pct=active.progress_pct,
                eta_seconds=active.eta_seconds,
            ).dict(),
        )

    background_tasks.add_task(
        run_task, task, bundle, settings, task_store, cancellation, s3_client, logger
    )

    return TaskAcceptedResponse(task_id=task.task_id, status=task.status, total_rows=task.total_rows)

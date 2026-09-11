"""
Приём заявки в общую очередь; GPU работает независимо от HTTP-запроса.
"""

from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import ErrorResponse, InferRequest, TaskAcceptedResponse
from app.deps import get_logger, get_models, get_s3_client, get_settings, get_task_store
from app.tasks.state import prefixes_overlap

router = APIRouter(tags=["inference"])


def validate_paths(request, settings) -> None:
    for path, bucket in (
        (request.s3_input_path, settings.s3.bucket_in),
        (request.s3_output_path, settings.s3.bucket_out),
    ):
        parsed = urlsplit(path)
        if (
            parsed.scheme != "s3"
            or parsed.netloc != bucket
            or not parsed.path.strip("/")
            or parsed.query
            or parsed.fragment
            or any(part in (".", "..", "") for part in parsed.path[1:].split("/"))
        ):
            raise HTTPException(422, "Ожидается S3-префикс внутри настроенного бакета")
    system_path = f"s3://{settings.s3.bucket_out}/{settings.task_store.state_key}"
    if prefixes_overlap(request.s3_output_path, system_path):
        raise HTTPException(422, "Выходной префикс пересекается с файлом очереди")
    if prefixes_overlap(request.s3_input_path, request.s3_output_path):
        raise HTTPException(422, "Входной и выходной префиксы не должны пересекаться")


@router.post(
    "/infer",
    response_model=TaskAcceptedResponse,
    status_code=202,
    summary="Поставить расчёт в очередь",
    responses={
        202: {"description": "Заявка сохранена или найдена ранее принятая попытка"},
        404: {"model": ErrorResponse, "description": "Модель или входной префикс не найдены"},
        409: {
            "model": ErrorResponse,
            "description": "Конфликт ключа, занятый выходной префикс или полная очередь",
        },
        422: {
            "model": ErrorResponse,
            "description": "Некорректные параметры, колонки или непустой выход",
        },
        503: {
            "model": ErrorResponse,
            "description": "S3 недоступен. Повторите запрос с прежним ключом",
        },
    },
)
def infer(
    request: InferRequest,
    settings=Depends(get_settings),
    models=Depends(get_models),
    store=Depends(get_task_store),
    s3_client=Depends(get_s3_client),
    logger=Depends(get_logger),
):
    """
    Принимает заявку независимо от занятости GPU текущего пода.

    До постановки проверяет модель, входные parquet и выходной префикс.
    Входные данные должны оставаться неизменными до завершения расчёта.
    Выход не должен содержать parquet или `_SUCCESS` и пересекаться с входом,
    системным файлом либо выходом незавершённой задачи.

    Одинаковый `idempotency_key` и параметры возвращают исходную задачу,
    в том числе завершённую, пока она хранится в истории. Ответ 202 означает
    принятие заявки, а не завершение расчёта. Новая задача имеет статус QUEUED.
    """
    # Нормализуем завершающий слеш до сравнения ключа идемпотентности.
    request.s3_input_path = request.s3_input_path.rstrip("/")
    request.s3_output_path = request.s3_output_path.rstrip("/")
    validate_paths(request, settings)
    existing = store.find_request(
        request.idempotency_key, request.model_name, request.s3_input_path, request.s3_output_path
    )
    if existing is not None:
        return existing
    if request.model_name not in models:
        raise HTTPException(404, f"Модель {request.model_name} не найдена")

    from app.models.validation import validate_input_parquet

    try:
        validation = validate_input_parquet(
            request.s3_input_path, models[request.model_name], s3_client
        )
    except FileNotFoundError as exc:
        logger.warn(f"Входной префикс не найден в S3: {request.s3_input_path}")
        raise HTTPException(404, "Входной префикс не найден в S3") from exc
    if not validation.is_valid:
        raise HTTPException(422, {"missing_columns": validation.missing_columns})
    if s3_client.prefix_has_results(request.s3_output_path):
        # Пока шла валидация, другой под мог принять и начать тот же запрос.
        existing = store.find_request(
            request.idempotency_key,
            request.model_name,
            request.s3_input_path,
            request.s3_output_path,
        )
        if existing is not None:
            return existing
        raise HTTPException(422, "Выходной префикс содержит parquet или _SUCCESS")
    task = store.enqueue(
        request.idempotency_key,
        request.model_name,
        request.s3_input_path,
        request.s3_output_path,
        validation.total_rows,
    )
    logger.info(
        f"Заявка {task.task_id} сохранена: модель={task.model_name}, статус={task.status.value}"
    )
    return task

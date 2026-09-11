"""
Контракт API для постановки задач, чтения статуса и отмены.
"""

from pydantic import BaseModel, Field

from app.tasks.state import TaskStatus


class InferRequest(BaseModel):
    idempotency_key: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Стабильный ключ одной попытки расчёта Airflow",
        examples=["credit_cards/run-1/infer/attempt-1"],
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Имя модели из configs/models.yaml",
        examples=["credit_cards"],
    )
    s3_input_path: str = Field(
        description="Входной префикс внутри S3_BUCKET_IN с подготовленными parquet",
        examples=["s3://input-bucket/data/run-1"],
    )
    s3_output_path: str = Field(
        description="Выходной префикс внутри S3_BUCKET_OUT без parquet и _SUCCESS",
        examples=["s3://output-bucket/results/run-1"],
    )


class TaskAcceptedResponse(BaseModel):
    task_id: str = Field(description="Идентификатор задачи для опроса статуса и отмены")
    pod_id: str | None = Field(description="Под-исполнитель, null до назначения")
    status: TaskStatus = Field(description="Текущий статус, у новой заявки QUEUED")
    total_rows: int = Field(description="Количество строк во входных данных")


class TaskStatusResponse(BaseModel):
    task_id: str = Field(description="Идентификатор задачи")
    model_name: str = Field(description="Имя модели из конфигурации")
    pod_id: str | None = Field(description="Назначенный под, null до захвата задачи")
    status: TaskStatus = Field(description="DONE, FAILED и ABORTED — финальные статусы")
    processed_rows: int = Field(
        description="Количество обработанных строк по сохранённому прогрессу"
    )
    total_rows: int = Field(description="Количество входных строк")
    progress_pct: float = Field(description="Прогресс в процентах. 100 ещё не означает DONE")
    eta_seconds: float | None = Field(description="Оценка оставшихся секунд, null если недоступна")
    error: str | None = Field(description="Причина ошибки или таймаута, null при отсутствии")
    heartbeat_at: float | None = Field(description="Последний heartbeat: Unix timestamp в секундах")
    executor_alive: bool = Field(description="Признак актуального heartbeat, не проверка процесса")
    s3_output_path: str = Field(description="Физический S3-префикс результата. Читать после DONE")
    created_at: float = Field(description="Создание задачи: Unix timestamp в секундах")
    started_at: float | None = Field(description="Начало расчёта: Unix timestamp, null до старта")
    finished_at: float | None = Field(description="Завершение: Unix timestamp, null до завершения")


class TaskAbortResponse(BaseModel):
    task_id: str = Field(description="Идентификатор задачи")
    status: TaskStatus = Field(description="Статус после запроса отмены. ABORTING требует ожидания")


class ActiveTaskSummary(BaseModel):
    task_id: str = Field(description="Идентификатор задачи")
    pod_id: str | None = Field(description="Под-исполнитель, null до назначения")
    model_name: str = Field(description="Имя модели")
    status: TaskStatus = Field(description="Один из незавершённых статусов")
    progress_pct: float = Field(description="Сохранённый прогресс в процентах")
    eta_seconds: float | None = Field(description="Оценка оставшихся секунд, если доступна")
    executor_alive: bool = Field(description="Признак актуального heartbeat исполнителя")


class ActiveTasksResponse(BaseModel):
    active_tasks: list[ActiveTaskSummary] = Field(description="Незавершённые задачи всех подов")


class ErrorResponse(BaseModel):
    detail: str | dict | list = Field(
        description="Сообщение ошибки, отсутствующие колонки или список ошибок валидации"
    )


class HealthResponse(BaseModel):
    status: str = Field(description="ok или unavailable", examples=["ok"])

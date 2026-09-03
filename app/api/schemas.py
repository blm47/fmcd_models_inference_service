"""
Pydantic-схемы запросов/ответов. model_name уже заложен в InferRequest
как задел на масштабирование (сейчас реестр содержит одну модель, но
контракт API не придётся менять при переходе к 7 моделям).
"""

from pydantic import BaseModel, Field

from app.tasks.state import TaskStatus


class S3PathNotFoundResponse(BaseModel):
    error: str = "s3_path_not_found"
    s3_path: str
    detail: str


class InferRequest(BaseModel):
    model_name: str = Field(..., description="Имя модели в реестре")
    s3_input_path: str = Field(..., description="s3://bucket/path/input/")
    s3_output_path: str = Field(..., description="s3://bucket/path/output/")


class ValidationErrorResponse(BaseModel):
    error: str = "missing_features"
    missing_columns: list[str]


class TaskAcceptedResponse(BaseModel):
    task_id: str
    pod_id: str = Field(..., description="Под, который взял задачу в работу")
    status: TaskStatus
    total_rows: int


class TaskBusyResponse(BaseModel):
    error: str = "task_already_running"
    task_id: str
    pod_id: str = Field(..., description="Под, на котором уже выполняется активная задача")
    status: TaskStatus
    progress_pct: float
    eta_seconds: float | None


class TaskStatusResponse(BaseModel):
    task_id: str
    model_name: str
    pod_id: str
    status: TaskStatus
    processed_rows: int
    total_rows: int
    progress_pct: float
    eta_seconds: float | None
    error: str | None


class TaskAbortResponse(BaseModel):
    task_id: str
    status: TaskStatus


class ActiveTaskSummary(BaseModel):
    task_id: str
    pod_id: str
    model_name: str
    status: TaskStatus
    progress_pct: float
    eta_seconds: float | None


class ActiveTasksResponse(BaseModel):
    """
    Диагностический эндпоинт: все активные задачи по всем подам сервиса
    (правило "1 активная задача на 1 под" - при N=3 подах здесь может
    быть до 3 записей одновременно).
    """

    active_tasks: list[ActiveTaskSummary]

"""Контракт API для постановки задач, чтения статуса и отмены."""

from pydantic import BaseModel, Field

from app.tasks.state import TaskStatus


class InferRequest(BaseModel):
    idempotency_key: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Стабильный ключ одной попытки расчёта Airflow",
    )
    model_name: str = Field(..., min_length=1)
    s3_input_path: str
    s3_output_path: str


class TaskAcceptedResponse(BaseModel):
    task_id: str
    pod_id: str | None
    status: TaskStatus
    total_rows: int


class TaskStatusResponse(BaseModel):
    task_id: str
    model_name: str
    pod_id: str | None
    status: TaskStatus
    processed_rows: int
    total_rows: int
    progress_pct: float
    eta_seconds: float | None
    error: str | None
    heartbeat_at: float | None
    executor_alive: bool
    s3_output_path: str
    created_at: float
    started_at: float | None
    finished_at: float | None


class TaskAbortResponse(BaseModel):
    task_id: str
    status: TaskStatus


class ActiveTaskSummary(BaseModel):
    task_id: str
    pod_id: str | None
    model_name: str
    status: TaskStatus
    progress_pct: float
    eta_seconds: float | None
    executor_alive: bool


class ActiveTasksResponse(BaseModel):
    active_tasks: list[ActiveTaskSummary]

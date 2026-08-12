"""
Pydantic-схемы запросов/ответов. model_name уже заложен в InferRequest
как задел на масштабирование (сейчас реестр содержит одну модель, но
контракт API не придётся менять при переходе к 7 моделям).
"""

from pydantic import BaseModel, Field

from app.tasks.state import TaskStatus


class InferRequest(BaseModel):
    model_name: str = Field(..., description="Имя модели в реестре (на v1 — единственная)")
    s3_input_path: str = Field(..., description="s3://bucket/path/input.parquet")
    s3_output_path: str = Field(..., description="s3://bucket/path/output.parquet")


class ValidationErrorResponse(BaseModel):
    error: str = "missing_features"
    missing_columns: list[str]


class TaskAcceptedResponse(BaseModel):
    task_id: str
    status: TaskStatus
    total_rows: int


class TaskBusyResponse(BaseModel):
    error: str = "task_already_running"
    task_id: str
    status: TaskStatus
    progress_pct: float
    eta_seconds: float | None


class TaskStatusResponse(BaseModel):
    task_id: str
    model_name: str
    status: TaskStatus
    processed_rows: int
    total_rows: int
    progress_pct: float
    eta_seconds: float | None
    error: str | None


class TaskAbortResponse(BaseModel):
    task_id: str
    status: TaskStatus

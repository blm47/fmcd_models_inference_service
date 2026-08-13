"""
Pydantic-схемы запросов/ответов. model_name уже заложен в InferRequest
как задел на масштабирование (сейчас реестр содержит одну модель, но
контракт API не придётся менять при переходе к 7 моделям).

s3_input_path/s3_output_path - это ПРЕФИКСЫ (папки), а не пути к одиночным
файлам:
  - s3_input_path: префикс, под которым Spark сохранил DataFrame
    (_SUCCESS + множество part-*.parquet).
  - s3_output_path: префикс, куда сервис запишет part-*.parquet чанками
    по chunk_size строк + свой _SUCCESS маркер по завершении.
"""

from pydantic import BaseModel, Field

from app.tasks.state import TaskStatus


class InferRequest(BaseModel):
    model_name: str = Field(..., description="Имя модели в реестре (на v1 - единственная)")
    s3_input_path: str = Field(
        ...,
        description="Префикс с входными part-*.parquet, напр. s3://bucket/path/input/",
    )
    s3_output_path: str = Field(
        ...,
        description="Префикс для выходных part-*.parquet, напр. s3://bucket/path/output/",
    )


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

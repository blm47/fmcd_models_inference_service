"""
Состояние задачи и статусы общей очереди.
"""

import time
from dataclasses import dataclass, field
from enum import StrEnum


class TaskStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"
    FINALIZING = "FINALIZING"


ACTIVE_STATUSES = (TaskStatus.RUNNING, TaskStatus.ABORTING, TaskStatus.FINALIZING)
TERMINAL_STATUSES = (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.ABORTED)


class QueueConflictError(ValueError):
    """
    Ключ запроса или выходной префикс уже занят другой задачей.
    """


@dataclass
class TaskState:
    task_id: str
    model_name: str
    s3_input_path: str
    s3_output_path: str
    status: TaskStatus
    total_rows: int | None
    calc_utilization: bool = False
    idempotency_key: str | None = None
    pod_id: str | None = None
    processed_rows: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    last_modified: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    inference_elapsed_sec: float = 0.0
    owner_id: str | None = None
    cancel_requested: bool = False

    @property
    def progress_pct(self) -> float:
        return round(100 * self.processed_rows / self.total_rows, 1) if self.total_rows else 0.0

    @property
    def eta_seconds(self) -> float | None:
        if self.total_rows is None or not self.processed_rows or self.status != TaskStatus.RUNNING:
            return None
        return round(
            max(0, self.total_rows - self.processed_rows)
            * self.inference_elapsed_sec
            / self.processed_rows,
            1,
        )

"""
Модель состояния задачи + потокобезопасное хранилище всех задач за время
жизни пода (in-memory dict).

Дизайн-решение: "текущая активная задача" не отдельная переменная, а
вычисляется как единственная запись со статусом RUNNING в TaskStore
(в системе может быть только 1 активная задача одновременно -
это гарантируется в TaskManager.try_start_task через lock).
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"


@dataclass
class TaskState:
    task_id: str
    model_name: str
    s3_input_path: str
    s3_output_path: str
    status: TaskStatus
    total_rows: int
    processed_rows: int = 0
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    inference_elapsed_sec: float = 0.0

    @property
    def progress_pct(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return round(100.0 * self.processed_rows / self.total_rows, 1)

    @property
    def eta_seconds(self) -> float | None:
        """
        Простое среднее: (elapsed_inference_time / processed_rows) * remaining_rows.
        None, пока не обработан ни один чанк (нет данных для оценки скорости).
        """
        if self.processed_rows == 0 or self.status != TaskStatus.RUNNING:
            return None

        remaining_rows = max(self.total_rows - self.processed_rows, 0)
        avg_sec_per_row = self.inference_elapsed_sec / self.processed_rows
        return round(avg_sec_per_row * remaining_rows, 1)


class TaskStore:
    """Потокобезопасное хранилище всех TaskState за время жизни пода."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskState] = {}

    def add(self, task: TaskState) -> None:
        with self._lock:
            self._tasks[task.task_id] = task

    def get(self, task_id: str) -> TaskState | None:
        with self._lock:
            return self._tasks.get(task_id)

    def get_active(self) -> TaskState | None:
        """Возвращает единственную задачу в статусе RUNNING/ABORTING, если есть."""
        with self._lock:
            for task in self._tasks.values():
                if task.status in (TaskStatus.RUNNING, TaskStatus.ABORTING):
                    return task
        return None

    def update_progress(self, task_id: str, processed_rows: int, inference_elapsed_sec: float) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.processed_rows = processed_rows
            task.inference_elapsed_sec = inference_elapsed_sec
            task.updated_at = time.time()

    def set_status(self, task_id: str, status: TaskStatus, error: str | None = None) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = status
            task.updated_at = time.time()
            if status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.ABORTED):
                task.finished_at = time.time()
            if error is not None:
                task.error = error

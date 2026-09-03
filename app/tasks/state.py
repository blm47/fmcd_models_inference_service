"""
Модель состояния задачи + TaskStore - фасад над выбранным
TaskStorageBackend (см. app/tasks/backends/).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from app.tasks.backends.base import TaskStorageBackend


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
    pod_id: str = "unknown"
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
    """
    Фасад над TaskStorageBackend. Публичный API идентичен старой
    in-memory реализации (v1), кроме get_active(pod_id) и
    try_add_if_no_active(task), которые теперь работают в разрезе
    конкретного pod_id (см. докстринг модуля, раздел v3).

    Разделение read/write по договорённости с бизнес-требованиями:
      - get()/get_active() -> backend.read_state() - без захвата lock,
        т.к. это read-only операции (в первую очередь GET /status).
      - add()/update_progress()/set_status() -> backend.mutate() -
        критическая секция read-modify-write, обязательно под lock'ом
        backend'а (при S3-backend'е - lease-lock).
    """

    def __init__(self, backend: TaskStorageBackend) -> None:
        self._backend = backend

    def add(self, task: TaskState) -> None:
        def _add(state: dict[str, TaskState]) -> dict[str, TaskState]:
            state[task.task_id] = task
            return state

        self._backend.mutate(_add)

    def get(self, task_id: str) -> TaskState | None:
        return self._backend.read_state().get(task_id)

    def get_active(self, pod_id: str | None = None) -> TaskState | None:
        """
        Возвращает активную (RUNNING/ABORTING) задачу.

        pod_id=None  - возвращает ЛЮБУЮ активную задачу в системе
                        (по любому поду) - полезно для общей диагностики.
        pod_id=<str> - возвращает активную задачу ТОЛЬКО на этом поде,
                        если она есть (используется в основном сценарии
                        "не более одной активной задачи на под").
        """
        for task in self._backend.read_state().values():
            if task.status not in (TaskStatus.RUNNING, TaskStatus.ABORTING):
                continue
            if pod_id is None or task.pod_id == pod_id:
                return task
        return None

    def get_all_active(self) -> list[TaskState]:
        """Все активные задачи по всем подам (для наблюдаемости/диагностики)."""
        return [
            task
            for task in self._backend.read_state().values()
            if task.status in (TaskStatus.RUNNING, TaskStatus.ABORTING)
        ]

    def update_progress(self, task_id: str, processed_rows: int, inference_elapsed_sec: float) -> None:
        def _update(state: dict[str, TaskState]) -> dict[str, TaskState]:
            task = state[task_id]
            task.processed_rows = processed_rows
            task.inference_elapsed_sec = inference_elapsed_sec
            task.updated_at = time.time()
            return state

        self._backend.mutate(_update)

    def set_status(self, task_id: str, status: TaskStatus, error: str | None = None) -> None:
        def _update(state: dict[str, TaskState]) -> dict[str, TaskState]:
            task = state[task_id]
            task.status = status
            task.updated_at = time.time()
            if status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.ABORTED):
                task.finished_at = time.time()
            if error is not None:
                task.error = error
            return state

        self._backend.mutate(_update)

    def try_add_if_no_active(self, task: TaskState) -> TaskState | None:
        """
        Атомарно (в рамках ОДНОГО backend.mutate()-вызова, т.е. под общим
        межподовым lock'ом): если на поде task.pod_id уже есть активная
        задача - возвращает её без изменений; иначе добавляет task и
        возвращает None.

        Проверка активности выполняется СТРОГО в разрезе pod_id: задача
        на другом поде не мешает взять новую задачу на текущем - правило
        "1 активная задача на 1 под", а не "1 активная задача на весь
        сервис".
        """
        result: dict[str, TaskState | None] = {"active": None}

        def _try_add(state: dict[str, TaskState]) -> dict[str, TaskState]:
            for existing in state.values():
                if existing.pod_id != task.pod_id:
                    continue
                if existing.status in (TaskStatus.RUNNING, TaskStatus.ABORTING):
                    result["active"] = existing
                    return state
            state[task.task_id] = task
            return state

        self._backend.mutate(_try_add)
        return result["active"]

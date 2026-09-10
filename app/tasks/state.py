"""
Модель состояния задачи + TaskStore - фасад над выбранным
TaskStorageBackend (см. app/tasks/backends/).
"""

from __future__ import annotations

import time
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum

from app.tasks.backends.base import TaskStorageBackend


class TaskStatus(str, Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    ABORTING = "ABORTING"
    ABORTED = "ABORTED"
    FINALIZING = "FINALIZING"


ACTIVE_STATUSES = (TaskStatus.RUNNING, TaskStatus.ABORTING, TaskStatus.FINALIZING)
TERMINAL_STATUSES = (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.ABORTED)


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
    owner_id: str | None = None
    heartbeat_at: float | None = None
    cancel_requested: bool = False

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
    """Операции над задачами под best-effort lock backend'а: read/modify/write.

    Чтение не записывает данные и не захватывает S3 lease. Heartbeat timeout
    означает потерю связи, а не доказанную гибель исполнителя: он влияет только
    на отбор задач в список активных.
    """

    def __init__(
        self,
        backend: TaskStorageBackend,
        heartbeat_interval_sec: float = 30.0,
        heartbeat_timeout_sec: float = 180.0,
    ) -> None:
        if not math.isfinite(heartbeat_interval_sec) or heartbeat_interval_sec <= 0:
            raise ValueError("heartbeat_interval_sec must be positive and finite")
        if not math.isfinite(heartbeat_timeout_sec) or heartbeat_timeout_sec <= heartbeat_interval_sec:
            raise ValueError("heartbeat_timeout_sec must exceed heartbeat_interval_sec")
        self._backend = backend
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.owner_id = uuid.uuid4().hex

    def _claim(self, task: TaskState) -> None:
        task.owner_id = self.owner_id
        task.heartbeat_at = time.time()

    def _check_owner(self, task: TaskState) -> None:
        if task.owner_id != self.owner_id:
            raise PermissionError(f"This process does not own task {task.task_id}")

    def executor_alive(self, task: TaskState) -> bool:
        last_seen = task.heartbeat_at if task.heartbeat_at is not None else task.updated_at
        return task.status in ACTIVE_STATUSES and time.time() - last_seen < self.heartbeat_timeout_sec

    def add(self, task: TaskState) -> None:
        def _add(state):
            if task.task_id in state:
                raise ValueError(f"Task {task.task_id} already exists")
            self._claim(task)
            state[task.task_id] = task
            return state

        self._backend.mutate(_add)

    def get(self, task_id: str) -> TaskState | None:
        return self._backend.read_state().get(task_id)

    def get_active(self, pod_id: str | None = None) -> TaskState | None:
        return next((task for task in self.get_all_active() if pod_id is None or task.pod_id == pod_id), None)

    def get_all_active(self) -> list[TaskState]:
        return [task for task in self._backend.read_state().values() if self.executor_alive(task)]

    def heartbeat(self, task_id: str) -> None:
        def _heartbeat(state):
            task = state[task_id]
            self._check_owner(task)
            if task.status in ACTIVE_STATUSES:
                task.heartbeat_at = time.time()
            return state

        self._backend.mutate(_heartbeat)

    def update_progress(self, task_id: str, processed_rows: int, inference_elapsed_sec: float) -> None:
        def _update(state):
            task = state[task_id]
            self._check_owner(task)
            if task.status in ACTIVE_STATUSES:
                task.processed_rows = processed_rows
                task.inference_elapsed_sec = inference_elapsed_sec
                task.updated_at = time.time()
                task.heartbeat_at = task.updated_at
            return state

        self._backend.mutate(_update)

    @staticmethod
    def _set_terminal(task: TaskState, status: TaskStatus, error: str | None = None) -> None:
        task.status = status
        task.updated_at = time.time()
        task.finished_at = task.updated_at
        task.cancel_requested = False
        task.error = error

    def set_status(self, task_id: str, status: TaskStatus, error: str | None = None) -> None:
        def _update(state):
            task = state[task_id]
            self._check_owner(task)
            if task.status in TERMINAL_STATUSES:
                return state
            if status in TERMINAL_STATUSES:
                if status == TaskStatus.DONE and task.status != TaskStatus.FINALIZING:
                    raise ValueError("Completion must be prepared before marking a task DONE")
                self._set_terminal(task, status, error)
            else:
                raise ValueError("Use request_abort or prepare_completion for nonterminal transitions")
            return state

        self._backend.mutate(_update)

    def request_abort(self, task_id: str) -> TaskState | None:
        def _abort(state):
            task = state.get(task_id)
            if task is not None and task.status in (TaskStatus.RUNNING, TaskStatus.ABORTING):
                task.cancel_requested = True
                task.status = TaskStatus.ABORTING
                # Запрос к API не подтверждает, что исполнитель ещё жив.
                # Поэтому updated_at и heartbeat_at здесь не обновляем.
            return state

        return self._backend.mutate(_abort).get(task_id)

    def cancellation_requested(self, task_id: str) -> bool:
        task = self.get(task_id)
        if task is None:
            raise RuntimeError(f"Task {task_id} disappeared from shared storage")
        self._check_owner(task)
        return task.cancel_requested or task.status == TaskStatus.ABORTING

    def prepare_completion(self, task_id: str, processed_rows: int, inference_elapsed_sec: float) -> bool:
        def _prepare(state):
            task = state[task_id]
            self._check_owner(task)
            if task.status in TERMINAL_STATUSES:
                return state
            task.processed_rows = processed_rows
            task.inference_elapsed_sec = inference_elapsed_sec
            task.updated_at = time.time()
            task.heartbeat_at = task.updated_at
            if task.cancel_requested or task.status == TaskStatus.ABORTING:
                self._set_terminal(task, TaskStatus.ABORTED)
            else:
                # После фиксации этого решения принимать abort уже поздно.
                task.status = TaskStatus.FINALIZING
            return state

        task = self._backend.mutate(_prepare)[task_id]
        return task.status == TaskStatus.FINALIZING

    def try_add_if_no_active(self, task: TaskState) -> TaskState | None:
        blocked_task_id = None

        def _add(state):
            nonlocal blocked_task_id
            for existing in state.values():
                if existing.pod_id != task.pod_id or existing.status not in ACTIVE_STATUSES:
                    continue
                # TaskManager отдельно удерживает слот работающего локального
                # worker, независимо от heartbeat и неудачных запросов запуска.
                if self.executor_alive(existing):
                    blocked_task_id = existing.task_id
                    return state
            if task.task_id in state:
                raise ValueError(f"Task {task.task_id} already exists")
            self._claim(task)
            state[task.task_id] = task
            return state

        state = self._backend.mutate(_add)
        if task.task_id in state:
            return None
        return state[blocked_task_id]

    def cleanup(self) -> None:
        self._backend.cleanup()

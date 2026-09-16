"""
Логика очереди, назначения исполнителей, отмены и таймаутов.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from app.core.config import TaskStoreConfig
from app.tasks.backends.base import TaskBackend
from app.tasks.types import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
)
from app.tasks.types import (
    QueueConflictError as QueueConflictError,
)
from app.tasks.types import (
    TaskState as TaskState,
)
from app.tasks.types import (
    TaskStatus as TaskStatus,
)


def prefixes_overlap(left: str, right: str) -> bool:
    """
    Проверяет пересечение выходных S3-префиксов для предотвращения конфликтов задач.

    Префиксы пересекаются, если совпадают или один вложен в другой по границе «/».
    Завершающие символы «/» игнорируются.
    """
    left, right = left.rstrip("/"), right.rstrip("/")
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


class TaskStore:
    """
    Управляет очередью задач через backend для API и исполнителей.

    Обеспечивает идемпотентность запросов, проверку конфликтов выходных S3-префиксов,
    назначение задач, отмену, таймауты и retention. Изменения выполняются через
    backend.mutate; owner_id определяет владельца задач для этого экземпляра.
    """

    def __init__(self, backend: TaskBackend, config: TaskStoreConfig, logger: Any):
        """
        Сохраняет backend, настройки и общий logger, создаёт идентификатор владельца.
        """
        self.backend = backend
        self.config = config
        self.logger = logger
        self.owner_id = uuid.uuid4().hex
        self._maintenance_lock = threading.Lock()
        self._next_cleanup = 0.0

    def initialize(self) -> None:
        """
        Инициализирует backend перед началом работы с очередью.
        """
        self.backend.initialize()

    def close(self) -> None:
        """
        Освобождает ресурсы backend при завершении работы.
        """
        self.backend.close()

    def get(self, task_id: str) -> TaskState | None:
        """
        Возвращает состояние задачи по task_id или None, если задача отсутствует.
        """
        return self.backend.get(task_id)

    def find_request(
        self,
        key: str,
        model: str,
        input_path: str,
        output_path: str,
        calc_utilization: bool = False,
    ) -> TaskState | None:
        """
        Находит задачу по ключу идемпотентности key или возвращает None.

        Проверяет совпадение модели, входного и выходного S3-путей и флага
        calc_utilization. При несовпадении параметров вызывает QueueConflictError.
        """
        task = self.backend.get_by_key(key)
        return self._find_request(
            {task.task_id: task} if task else {},
            key,
            model,
            input_path,
            output_path,
            calc_utilization,
        )

    @staticmethod
    def _find_request(tasks, key, model, input_path, output_path, calc_utilization=False):
        """
        Ищет запрос в tasks; конфликт параметров вызывает QueueConflictError.

        Возвращает найденную задачу или None, если ключ ещё не использован.
        """
        for task in tasks.values():
            if task.idempotency_key == key:
                if (
                    task.model_name,
                    task.s3_input_path,
                    task.s3_output_path,
                    task.calc_utilization,
                ) != (
                    model,
                    input_path,
                    output_path,
                    calc_utilization,
                ):
                    raise QueueConflictError(
                        "Ключ идемпотентности уже использован с другими параметрами"
                    )
                return task
        return None

    def enqueue(
        self,
        key: str,
        model: str,
        input_path: str,
        output_path: str,
        total_rows: int | None,
        calc_utilization: bool = False,
    ) -> TaskState:
        """
        Атомарно ставит задачу в QUEUED или возвращает идентичный повторный запрос.

        key задаёт ключ идемпотентности, model — имя модели, input_path и
        output_path — S3-пути. total_rows содержит число входных строк либо None,
        если оно ещё неизвестно; calc_utilization включает расчёт utilization.

        Вызывает QueueConflictError при повторном ключе с другими параметрами,
        пересечении выходного префикса с незавершённой задачей или заполнении очереди.
        """
        task_id = str(uuid.uuid4())

        def add(tasks):
            existing = self._find_request(
                tasks, key, model, input_path, output_path, calc_utilization
            )
            if existing:
                return existing
            pending = [task for task in tasks.values() if task.status not in TERMINAL_STATUSES]
            if any(prefixes_overlap(task.s3_output_path, output_path) for task in pending):
                raise QueueConflictError("Выходной префикс занят незавершённой задачей")
            if len(pending) >= self.config.max_pending_tasks:
                raise QueueConflictError("Очередь заполнена")
            task = TaskState(
                task_id=task_id,
                idempotency_key=key,
                model_name=model,
                s3_input_path=input_path,
                s3_output_path=output_path,
                status=TaskStatus.QUEUED,
                total_rows=total_rows,
                calc_utilization=calc_utilization,
            )
            tasks[task_id] = task
            return task

        return self.backend.mutate(add, idempotency_key=key)

    def claim_next(self, pod_id: str, model_names: set[str]) -> TaskState | None:
        """
        Назначает pod_id самую раннюю задачу из очереди для моделей model_names.

        Переводит её в RUNNING и сохраняет владельца и время старта. Сначала
        возвращает уже активную задачу этого владельца, если она есть.
        Если подходящих задач нет, возвращает None.
        """
        def claim(tasks):
            # Восстанавливаем результат захвата, если ответ предыдущей записи потерялся.
            owned = next(
                (
                    task
                    for task in tasks.values()
                    if task.owner_id == self.owner_id and task.status in ACTIVE_STATUSES
                ),
                None,
            )
            if owned:
                return owned
            candidates = [
                task
                for task in tasks.values()
                if task.status == TaskStatus.QUEUED and task.model_name in model_names
            ]
            if not candidates:
                return None
            task = min(candidates, key=lambda item: (item.created_at, item.task_id))
            task.status = TaskStatus.RUNNING
            task.pod_id = pod_id
            task.owner_id = self.owner_id
            task.started_at = task.last_modified = time.time()
            return task

        return self.backend.mutate(claim)

    def executor_alive(self, task: TaskState) -> bool:
        """
        Проверяет активный статус и отсутствие таймаута без изменения задачи.
        """
        return task.status in ACTIVE_STATUSES and self._timeout_reason(task) is None

    def get_all_active(self) -> list[TaskState]:
        """
        Возвращает список активных задач из backend.
        """
        return self.backend.list_active()

    def _check_owner(self, task: TaskState) -> None:
        """
        Вызывает PermissionError, если задача принадлежит другому владельцу.
        """
        if task.owner_id != self.owner_id:
            raise PermissionError(f"Процесс не владеет задачей {task.task_id}")

    def heartbeat(self, task_id: str) -> bool:
        """
        Подтверждает активность исполнителя, не оживляя просроченную задачу.

        Возвращает True при обновлении времени активности, иначе False.
        Для чужой задачи вызывает PermissionError, для отсутствующей — KeyError.
        """

        def update(tasks):
            task = tasks[task_id]
            self._check_owner(task)
            self._expire_task(task)
            if task.status not in ACTIVE_STATUSES:
                return False
            task.last_modified = time.time()
            return True

        return self.backend.mutate(update, task_id=task_id)

    def set_total_rows(self, task_id: str, total_rows: int) -> bool:
        """
        Сохраняет результат проверки входа и время обновления исполнителем.

        Возвращает True при обновлении задачи в RUNNING, иначе False.
        Вызывает ValueError, если total_rows не является неотрицательным int,
        PermissionError для чужой задачи и KeyError для отсутствующей.
        """
        if type(total_rows) is not int or total_rows < 0:
            raise ValueError("total_rows: ожидается неотрицательное целое число")

        def update(tasks):
            task = tasks[task_id]
            self._check_owner(task)
            self._expire_task(task)
            if task.status != TaskStatus.RUNNING:
                return False
            task.total_rows = total_rows
            task.last_modified = time.time()
            return True

        return self.backend.mutate(update, task_id=task_id)

    def update_progress(
        self, task_id: str, processed_rows: int, inference_elapsed_sec: float
    ) -> None:
        """
        Обновляет число обработанных строк и время inference в секундах.

        Сначала применяет таймаут; прогресс сохраняется только для активной задачи.
        Для чужой задачи вызывает PermissionError, для отсутствующей — KeyError.
        """
        def update(tasks):
            task = tasks[task_id]
            self._check_owner(task)
            self._expire_task(task)
            if task.status in ACTIVE_STATUSES:
                task.processed_rows = processed_rows
                task.inference_elapsed_sec = inference_elapsed_sec
                task.last_modified = time.time()

        self.backend.mutate(update, task_id=task_id)

    @staticmethod
    def _set_terminal(task: TaskState, status: TaskStatus, error: str | None = None) -> None:
        """
        Задаёт финальный статус, время завершения и ошибку, сбрасывает флаг отмены.
        """
        task.status = status
        task.last_modified = task.finished_at = time.time()
        task.cancel_requested = False
        task.error = error

    def set_status(self, task_id: str, status: TaskStatus, error: str | None = None) -> bool:
        """
        Завершает задачу владельца с указанными финальным статусом и ошибкой.

        Возвращает True, если итоговый статус совпадает с запрошенным; уже
        завершённую задачу не изменяет. Для незавершённой задачи вызывает ValueError
        при нефинальном статусе или переходе в DONE без FINALIZING.
        Для чужой задачи вызывает PermissionError, для отсутствующей — KeyError.
        """
        def update(tasks):
            task = tasks[task_id]
            self._check_owner(task)
            self._expire_task(task)
            if task.status in TERMINAL_STATUSES:
                return task.status == status
            if status not in TERMINAL_STATUSES:
                raise ValueError("Ожидается финальный статус")
            if status == TaskStatus.DONE and task.status != TaskStatus.FINALIZING:
                raise ValueError("Перед DONE требуется FINALIZING")
            self._set_terminal(task, status, error)
            return True

        return self.backend.mutate(update, task_id=task_id)

    def request_abort(self, task_id: str) -> TaskState | None:
        """
        Запрашивает отмену из API и возвращает задачу либо None, если её нет.

        Задачу в QUEUED сразу переводит в ABORTED; для RUNNING и ABORTING
        устанавливает флаг отмены и статус ABORTING без обновления heartbeat.
        Для FINALIZING вызывает QueueConflictError, финальные статусы не меняет.
        """
        def abort(tasks):
            task = tasks.get(task_id)
            if task is not None:
                if task.status == TaskStatus.QUEUED:
                    self._set_terminal(task, TaskStatus.ABORTED)
                elif task.status in (TaskStatus.RUNNING, TaskStatus.ABORTING):
                    # Запрос API не подтверждает активность исполнителя.
                    task.cancel_requested = True
                    task.status = TaskStatus.ABORTING
                elif task.status == TaskStatus.FINALIZING:
                    raise QueueConflictError(
                        "Публикация результата уже началась; отмена недоступна"
                    )
            return task

        return self.backend.mutate(abort, task_id=task_id)

    def cancellation_requested(self, task_id: str) -> bool:
        """
        Возвращает признак отмены активной задачи текущего владельца.

        Вызывает RuntimeError, если задача отсутствует, завершена или просрочена,
        и PermissionError, если она принадлежит другому владельцу.
        """
        task = self.get(task_id)
        if task is None:
            raise RuntimeError(f"Задача {task_id} отсутствует в очереди")
        self._check_owner(task)
        if task.status not in ACTIVE_STATUSES or self._timeout_reason(task):
            raise RuntimeError(f"task_not_active: задача {task_id} завершена или просрочена")
        return task.cancel_requested or task.status == TaskStatus.ABORTING

    def prepare_completion(
        self, task_id: str, processed_rows: int, inference_elapsed_sec: float
    ) -> bool:
        """
        Сохраняет итоговый прогресс и готовит задачу к публикации результата.

        После проверки таймаута переводит незавершённую задачу в ABORTED при
        запросе отмены, иначе в FINALIZING. Возвращает True только для FINALIZING.
        Для чужой задачи вызывает PermissionError, для отсутствующей — KeyError.
        """
        def prepare(tasks):
            task = tasks[task_id]
            self._check_owner(task)
            self._expire_task(task)
            if task.status not in TERMINAL_STATUSES:
                task.processed_rows = processed_rows
                task.inference_elapsed_sec = inference_elapsed_sec
                task.last_modified = time.time()
                if task.cancel_requested:
                    self._set_terminal(task, TaskStatus.ABORTED)
                else:
                    task.status = TaskStatus.FINALIZING
            return task.status == TaskStatus.FINALIZING

        return self.backend.mutate(prepare, task_id=task_id)

    def _timeout_reason(self, task: TaskState) -> str | None:
        """
        Возвращает причину таймаута активной задачи или None без изменения состояния.
        """
        if task.status not in ACTIVE_STATUSES:
            return None
        if time.time() - task.last_modified >= self.config.task_timeout_sec:
            return "task_timeout: исполнитель не обновляет задачу"
        return None

    def _expire_task(self, task: TaskState) -> bool:
        """
        Переводит просроченную активную задачу в FAILED и возвращает признак изменения.
        """
        reason = self._timeout_reason(task)
        if reason is None:
            return False
        self._set_terminal(task, TaskStatus.FAILED, reason)
        return True

    def fail_stale(self) -> None:
        """
        Атомарно завершает просроченные расчёты, не затрагивая очередь и финальные статусы.
        """

        def expire(tasks):
            return [task.task_id for task in tasks.values() if self._expire_task(task)]

        for task_id in self.backend.mutate(expire):
            self.logger.warn(f"Задача {task_id} переведена в FAILED по таймауту")

    def check_queue(self) -> bool:
        """
        Readiness проверяет таймауты и запускает retention по расписанию без потока.

        Одновременные запросы не накапливают проверки при медленном хранилище.
        Возвращает True после обслуживания или False, если оно уже выполняется.
        """
        if not self._maintenance_lock.acquire(blocking=False):
            return False
        try:
            self.fail_stale()
            if time.monotonic() >= self._next_cleanup:
                self.cleanup()
                self._next_cleanup = time.monotonic() + self.config.cleanup_interval_sec
            return True
        finally:
            self._maintenance_lock.release()

    def cleanup(self) -> None:
        """
        Запускает очистку backend по retention, если backend её поддерживает.

        Порог хранения вычисляется из retention_months, считая месяц равным 30 дням.
        """
        if self.backend.supports_retention:
            cutoff = time.time() - self.config.retention_months * 30 * 24 * 60 * 60
            self.backend.cleanup(cutoff)

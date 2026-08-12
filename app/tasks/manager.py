"""
TaskManager — единственная точка входа для запуска/проверки/отмены задач.

Ключевое правило архитектуры: "новая задача берётся в работу, только если
нет активной; иначе — отбивка с ИД активной задачи, прогрессом и ETA".
Это гарантируется атомарной проверкой-и-вставкой под одним lock'ом
(TaskStore.get_active + add происходят как одна критическая секция здесь,
а не по отдельности, чтобы избежать гонки между двумя параллельными
запросами /infer, прилетевшими почти одновременно).
"""

import threading
import uuid

from app.tasks.cancellation import CancellationRegistry
from app.tasks.state import TaskState, TaskStatus, TaskStore


class TaskAlreadyRunningError(Exception):
    def __init__(self, active_task: TaskState):
        self.active_task = active_task
        super().__init__(f"Задача {active_task.task_id} уже выполняется")


class TaskManager:
    def __init__(self, store: TaskStore, cancellation: CancellationRegistry):
        self._store = store
        self._cancellation = cancellation
        self._start_lock = threading.Lock()

    def try_start_task(
        self, model_name: str, s3_input_path: str, s3_output_path: str, total_rows: int
    ) -> TaskState:
        """
        Атомарно: если есть активная задача -> кидает TaskAlreadyRunningError
        с её состоянием (для формирования 409-ответа с прогрессом/ETA).
        Иначе создаёт новую TaskState со статусом RUNNING и регистрирует
        cancellation-флаг, возвращает её вызывающему коду (роуту), который
        должен передать task_id в BackgroundTasks.
        """
        with self._start_lock:
            active = self._store.get_active()
            if active is not None:
                raise TaskAlreadyRunningError(active)

            task_id = str(uuid.uuid4())
            task = TaskState(
                task_id=task_id,
                model_name=model_name,
                s3_input_path=s3_input_path,
                s3_output_path=s3_output_path,
                status=TaskStatus.RUNNING,
                total_rows=total_rows,
            )
            self._store.add(task)
            self._cancellation.create(task_id)
            return task

    def get_status(self, task_id: str) -> TaskState | None:
        return self._store.get(task_id)

    def request_abort(self, task_id: str) -> TaskState | None:
        task = self._store.get(task_id)
        if task is None or task.status not in (TaskStatus.RUNNING,):
            return task
        self._cancellation.request_cancel(task_id)
        self._store.set_status(task_id, TaskStatus.ABORTING)
        return self._store.get(task_id)

"""
TaskManager - единственная точка входа для запуска/проверки/отмены задач.

Правило активности задачи - "не более одной
активной задачи на 1 под" (Если сервис работает на N=3 подах, значит суммарно
допускается до 3 параллельных фоновых инференс-задач - по одной на под).
pod_id определяется самим процессом при старте
и передаётся в TaskManager снаружи (из lifespan/deps).

Если на ТЕКУЩЕМ поде уже есть активная задача - отбой с
TaskAlreadyRunningError (409 в роуте), тем же контрактом, что и раньше.
Если активна задача на ДРУГОМ поде, а на текущем свободно - новая задача
берётся в работу.
"""

import threading
import uuid

from app.tasks.cancellation import CancellationRegistry
from app.tasks.state import TaskState, TaskStatus, TaskStore


class TaskAlreadyRunningError(Exception):
    def __init__(self, active_task: TaskState):
        self.active_task = active_task
        super().__init__(
            f"На поде {active_task.pod_id} уже выполняется задача {active_task.task_id}"
        )


class TaskManager:
    def __init__(self, store: TaskStore, cancellation: CancellationRegistry, pod_id: str):
        self._store = store
        self._cancellation = cancellation
        self._pod_id = pod_id
        self._start_lock = threading.Lock()

    @property
    def pod_id(self) -> str:
        return self._pod_id

    def try_start_task(
        self, model_name: str, s3_input_path: str, s3_output_path: str, total_rows: int
    ) -> TaskState:
        """
        Атомарно (межподово, через backend.mutate(), в разрезе pod_id):
        если на ТЕКУЩЕМ поде есть активная задача -> кидает
        TaskAlreadyRunningError с её состоянием (409-ответ с task_id,
        статусом и ETA той задачи, что занимает именно этот под). Если
        активна задача только на ДРУГОМ поде - берёт новую задачу на
        текущем поде без проблем. threading.Lock здесь защищает только от
        гонки внутри одного процесса (два почти одновременных HTTP-запроса
        на один под); межподовая гонка закрывается backend.mutate().
        """
        with self._start_lock:
            task_id = str(uuid.uuid4())
            task = TaskState(
                task_id=task_id,
                model_name=model_name,
                s3_input_path=s3_input_path,
                s3_output_path=s3_output_path,
                status=TaskStatus.RUNNING,
                total_rows=total_rows,
                pod_id=self._pod_id,
            )
            active_on_this_pod = self._store.try_add_if_no_active(task)
            if active_on_this_pod is not None:
                raise TaskAlreadyRunningError(active_on_this_pod)
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

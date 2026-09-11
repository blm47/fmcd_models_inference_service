"""
Общая очередь в одном объекте S3. Все изменения выполняются через CAS.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from botocore.exceptions import ClientError, ConnectionError, HTTPClientError

from app.core.config import TaskStoreConfig

T = TypeVar("T")


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
    """Ключ запроса или выходной префикс уже занят другой задачей."""


@dataclass
class TaskState:
    task_id: str
    model_name: str
    s3_input_path: str
    s3_output_path: str
    status: TaskStatus
    total_rows: int
    idempotency_key: str | None = None
    pod_id: str | None = None
    processed_rows: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str | None = None
    inference_elapsed_sec: float = 0.0
    owner_id: str | None = None
    heartbeat_at: float | None = None
    cancel_requested: bool = False

    @property
    def progress_pct(self) -> float:
        return round(100 * self.processed_rows / self.total_rows, 1) if self.total_rows else 0.0

    @property
    def eta_seconds(self) -> float | None:
        if not self.processed_rows or self.status != TaskStatus.RUNNING:
            return None
        return round(
            max(0, self.total_rows - self.processed_rows)
            * self.inference_elapsed_sec
            / self.processed_rows,
            1,
        )


def prefixes_overlap(left: str, right: str) -> bool:
    left, right = left.rstrip("/"), right.rstrip("/")
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


class TaskStore:
    """
    Читает очередь, принимает заявки и назначает исполнителей.
    """

    def __init__(self, client, bucket: str, config: TaskStoreConfig, logger: Any):
        self.client = client
        self.bucket = bucket
        self.config = config
        self.logger = logger
        # Идентификатор процесса не переиспользуется после рестарта пода.
        self.owner_id = uuid.uuid4().hex
        self.heartbeat_interval_sec = config.heartbeat_interval_sec

    @staticmethod
    def _encode(tasks: dict[str, TaskState], revision: int) -> bytes:
        return json.dumps(
            {"revision": revision, "tasks": {key: asdict(task) for key, task in tasks.items()}},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")

    def _read(self) -> tuple[dict[str, TaskState], str, int]:
        response = self.client.get_object(Bucket=self.bucket, Key=self.config.state_key)
        body = response["Body"]
        try:
            raw = json.loads(body.read())
        finally:
            body.close()
        # Пустой или повреждённый объект не превращаем в пустую очередь.
        revision = raw.get("revision", 0)
        if type(revision) is not int or revision < 0:
            raise ValueError("Некорректная revision в очереди S3")
        tasks = {}
        for key, fields in raw["tasks"].items():
            # Старые записи не содержат created_at; сохраняем стабильное время при миграции.
            fields = dict(fields)
            fields.setdefault("created_at", fields.get("started_at") or fields["updated_at"])
            task = TaskState(**{**fields, "status": TaskStatus(fields["status"])})
            if key != task.task_id:
                raise ValueError("Ключ задачи не совпадает с task_id")
            tasks[key] = task
        etag = response["ETag"]
        if self.config.strip_etag_quotes:
            etag = etag.strip('"')
        return tasks, etag, revision

    def initialize(self) -> None:
        """
        Создаёт объект только при отсутствии; одновременный старт подов допустим.
        """
        deadline = time.monotonic() + self.config.cas_timeout_sec
        while True:
            try:
                self._read()
                return
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in ("NoSuchKey", "404"):
                    raise
            try:
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=self.config.state_key,
                    Body=self._encode({}, 0),
                    ContentType="application/json",
                    IfNoneMatch="*",
                )
                self.logger.info(f"Создан файл очереди S3: {self.config.state_key}")
                return
            except (ClientError, ConnectionError, HTTPClientError) as exc:
                if isinstance(exc, ClientError) and not self._retryable(exc):
                    raise
                self._pause(deadline)

    @staticmethod
    def _retryable(exc: ClientError) -> bool:
        status = exc.response["ResponseMetadata"]["HTTPStatusCode"]
        return status in (409, 412, 429) or status >= 500

    def _pause(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Истёк срок обновления очереди S3; повторите запрос с тем же ключом")
        time.sleep(min(remaining, random.uniform(0.5, 1.5) * self.config.cas_retry_interval_sec))

    def _mutate(self, change: Callable[[dict[str, TaskState]], T]) -> T:
        """
        Повторяет чистое идемпотентное изменение после конфликта или потери ответа PUT.
        """
        deadline = time.monotonic() + self.config.cas_timeout_sec
        while True:
            try:
                tasks, etag, revision = self._read()
                before = self._encode(tasks, revision)
                result = change(tasks)
                if self._encode(tasks, revision) == before:
                    return result
                self.client.put_object(
                    Bucket=self.bucket,
                    Key=self.config.state_key,
                    Body=self._encode(tasks, revision + 1),
                    ContentType="application/json",
                    IfMatch=etag,
                )
                return result
            except (ClientError, ConnectionError, HTTPClientError) as exc:
                if isinstance(exc, ClientError) and not self._retryable(exc):
                    raise
                # GET после неопределённого PUT позволяет увидеть уже сохранённый результат.
                self._pause(deadline)

    def get(self, task_id: str) -> TaskState | None:
        return self._read()[0].get(task_id)

    def find_request(
        self, key: str, model: str, input_path: str, output_path: str
    ) -> TaskState | None:
        return self._find_request(self._read()[0], key, model, input_path, output_path)

    @staticmethod
    def _find_request(tasks, key, model, input_path, output_path):
        for task in tasks.values():
            if task.idempotency_key == key:
                if (task.model_name, task.s3_input_path, task.s3_output_path) != (
                    model,
                    input_path,
                    output_path,
                ):
                    raise QueueConflictError(
                        "Ключ идемпотентности уже использован с другими параметрами"
                    )
                return task
        return None

    def enqueue(
        self, key: str, model: str, input_path: str, output_path: str, total_rows: int
    ) -> TaskState:
        task_id = str(uuid.uuid4())

        def add(tasks):
            existing = self._find_request(tasks, key, model, input_path, output_path)
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
            )
            tasks[task_id] = task
            return task

        return self._mutate(add)

    def claim_next(self, pod_id: str, model_names: set[str]) -> TaskState | None:
        def claim(tasks):
            # Восстанавливаем результат захвата, если ответ предыдущего PUT потерялся.
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
            task.started_at = task.updated_at = task.heartbeat_at = time.time()
            return task

        return self._mutate(claim)

    def executor_alive(self, task: TaskState) -> bool:
        last_seen = task.heartbeat_at if task.heartbeat_at is not None else task.updated_at
        return (
            task.status in ACTIVE_STATUSES
            and time.time() - last_seen < self.config.heartbeat_timeout_sec
        )

    def get_all_active(self) -> list[TaskState]:
        # Просроченный heartbeat не скрывает незавершённый расчёт.
        return [task for task in self._read()[0].values() if task.status not in TERMINAL_STATUSES]

    def _check_owner(self, task: TaskState) -> None:
        if task.owner_id != self.owner_id:
            raise PermissionError(f"Процесс не владеет задачей {task.task_id}")

    def heartbeat(self, task_id: str) -> bool:
        def update(tasks):
            task = tasks[task_id]
            self._check_owner(task)
            self._expire_task(task)
            if task.status in ACTIVE_STATUSES:
                task.heartbeat_at = time.time()
                return True
            return False

        return self._mutate(update)

    def update_progress(
        self, task_id: str, processed_rows: int, inference_elapsed_sec: float
    ) -> None:
        def update(tasks):
            task = tasks[task_id]
            self._check_owner(task)
            self._expire_task(task)
            if task.status in ACTIVE_STATUSES:
                task.processed_rows = processed_rows
                task.inference_elapsed_sec = inference_elapsed_sec
                task.updated_at = task.heartbeat_at = time.time()

        self._mutate(update)

    @staticmethod
    def _set_terminal(task: TaskState, status: TaskStatus, error: str | None = None) -> None:
        task.status = status
        task.updated_at = task.finished_at = time.time()
        task.cancel_requested = False
        task.error = error

    def set_status(self, task_id: str, status: TaskStatus, error: str | None = None) -> bool:
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

        return self._mutate(update)

    def request_abort(self, task_id: str) -> TaskState | None:
        def abort(tasks):
            task = tasks.get(task_id)
            if task is not None:
                if task.status == TaskStatus.QUEUED:
                    self._set_terminal(task, TaskStatus.ABORTED)
                elif task.status in (TaskStatus.RUNNING, TaskStatus.ABORTING):
                    task.cancel_requested = True
                    task.status = TaskStatus.ABORTING
                elif task.status == TaskStatus.FINALIZING:
                    raise QueueConflictError(
                        "Публикация результата уже началась; отмена недоступна"
                    )
            return task

        return self._mutate(abort)

    def cancellation_requested(self, task_id: str) -> bool:
        task = self.get(task_id)
        if task is None:
            raise RuntimeError(f"Задача {task_id} отсутствует в очереди S3")
        self._check_owner(task)
        if task.status not in ACTIVE_STATUSES or self._timeout_reason(task):
            raise RuntimeError(f"task_not_active: задача {task_id} завершена или просрочена")
        return task.cancel_requested or task.status == TaskStatus.ABORTING

    def prepare_completion(
        self, task_id: str, processed_rows: int, inference_elapsed_sec: float
    ) -> bool:
        def prepare(tasks):
            task = tasks[task_id]
            self._check_owner(task)
            self._expire_task(task)
            if task.status not in TERMINAL_STATUSES:
                task.processed_rows = processed_rows
                task.inference_elapsed_sec = inference_elapsed_sec
                task.updated_at = task.heartbeat_at = time.time()
                if task.cancel_requested:
                    self._set_terminal(task, TaskStatus.ABORTED)
                else:
                    task.status = TaskStatus.FINALIZING
            return task.status == TaskStatus.FINALIZING

        return self._mutate(prepare)

    def _timeout_reason(self, task: TaskState) -> str | None:
        if task.status not in ACTIVE_STATUSES:
            return None
        now = time.time()
        last_heartbeat = task.heartbeat_at if task.heartbeat_at is not None else task.updated_at
        if now - last_heartbeat >= self.config.heartbeat_timeout_sec:
            return "heartbeat_timeout: исполнитель не обновляет heartbeat"
        if now - task.updated_at >= self.config.progress_timeout_sec:
            return "progress_timeout: исполнитель не обновляет прогресс"
        return None

    def _expire_task(self, task: TaskState) -> bool:
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

        for task_id in self._mutate(expire):
            self.logger.warn(f"Задача {task_id} переведена в FAILED по таймауту")

    def cleanup(self) -> None:
        cutoff = time.time() - self.config.retention_months * 30 * 24 * 60 * 60

        def clean(tasks):
            expired = [
                key
                for key, task in tasks.items()
                if task.status in TERMINAL_STATUSES
                and (task.finished_at if task.finished_at is not None else task.updated_at) < cutoff
            ]
            for key in expired:
                del tasks[key]

        self._mutate(clean)

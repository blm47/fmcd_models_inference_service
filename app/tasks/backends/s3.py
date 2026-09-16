"""
Очередь в одном объекте S3 с атомарным CAS по ETag.
"""

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import TypeVar

from botocore.exceptions import ClientError, ConnectionError, HTTPClientError

from app.tasks.types import TERMINAL_STATUSES, TaskState, TaskStatus

T = TypeVar("T")


class S3TaskBackend:
    supports_retention = True

    def __init__(self, client, bucket, config, logger):
        self.client = client
        self.bucket = bucket
        self.config = config
        self.logger = logger

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

    def mutate(
        self, change: Callable[[dict[str, TaskState]], T], *, task_id=None, idempotency_key=None
    ) -> T:
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

    def get(self, task_id):
        return self._read()[0].get(task_id)

    def get_by_key(self, key):
        return next(
            (task for task in self._read()[0].values() if task.idempotency_key == key), None
        )

    def list_active(self):
        return [task for task in self._read()[0].values() if task.status not in TERMINAL_STATUSES]

    def cleanup(self, cutoff):
        def clean(tasks):
            expired = [
                key
                for key, task in tasks.items()
                if task.status in TERMINAL_STATUSES
                and (task.finished_at or task.last_modified) < cutoff
            ]
            for key in expired:
                del tasks[key]

        self.mutate(clean)

    def close(self):
        self.client.close()

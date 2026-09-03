"""
S3-реализация TaskStorageBackend через lease-lock (без DELETE).

Контекст ограничения: у сервисного технического пользователя (ТУЗа) в S3
есть права только на чтение и запись/перезапись объекта - НЕТ прав на
удаление. Поэтому классическая схема "lock = существование файла,
разблокировать = удалить файл" здесь невозможна.

Решение: lock - это не факт существования объекта,
а LEASE с полем expires_at внутри JSON-содержимого самого lock-объекта.
  - "Лок захвачен и активен"  <=>  lock.expires_at > now()
  - "Лок свободен/протух"     <=>  lock существует, но expires_at <= now(),
    либо объект ещё не создавался.
  - "Освободить лок"  =  перезаписать lock-объект с истёкшим/нулевым
    expires_at (НЕ удаление объекта - прав на delete нет).
  - "Лок мёртв, хотя формально не протух" (stale) - защита от пода,
    который держал lock, начал read-modify-write и упал / был убит до
    того, как успел его продлить или снять.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from typing import Any

from app.tasks.backends.base import MutateFn, TaskStorageBackend
from app.tasks.state import TaskState, TaskStatus

DEFAULT_LEASE_SECONDS = 20
DEFAULT_WAIT_TIMEOUT_SECONDS = 60
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_RETENTION_MONTHS = 6

_SECONDS_PER_MONTH = 30 * 24 * 60 * 60  # приближение, месяц = 30 суток


class LockAcquireTimeoutError(Exception):
    """Не удалось захватить lease-lock за wait_timeout_sec."""


class S3LeaseLockBackend(TaskStorageBackend):
    def __init__(
        self,
        s3_fs: Any,
        state_key: str,
        lock_key: str,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        wait_timeout_sec: int = DEFAULT_WAIT_TIMEOUT_SECONDS,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SECONDS,
        retention_months: int = DEFAULT_RETENTION_MONTHS,
    ) -> None:
        self._fs = s3_fs
        self._state_key = state_key
        self._lock_key = lock_key
        self._lease_seconds = lease_seconds
        self._wait_timeout_sec = wait_timeout_sec
        self._poll_interval_sec = poll_interval_sec
        self._retention_months = retention_months
        self._owner_id = uuid.uuid4().hex

    def read_state(self) -> dict[str, TaskState]:
        raw = self._read_json(self._state_key)
        if raw is None:
            return {}
        return self._deserialize_state(raw)

    def mutate(self, fn: MutateFn) -> dict[str, TaskState]:
        self._wait_and_acquire_lock()
        try:
            state = self.read_state()
            new_state = fn(state)
            new_state = self._apply_retention(new_state)
            self._write_json(self._state_key, self._serialize_state(new_state))
            return new_state
        finally:
            self._release_lock()

    def _read_lock(self) -> dict[str, Any] | None:
        return self._read_json(self._lock_key)

    def _lock_is_active(self, lock: dict[str, Any] | None) -> bool:
        if lock is None:
            return False
        expires_at = lock.get("expires_at", 0.0)
        return time.time() < expires_at

    def _active_lock_is_stale(self, lock: dict[str, Any] | None) -> bool:
        if lock is None:
            return False
        acquired_at = lock.get("acquired_at", 0.0)
        return time.time() - acquired_at > 2 * self._lease_seconds

    def _wait_and_acquire_lock(self) -> None:
        deadline = time.time() + self._wait_timeout_sec
        while True:
            lock = self._read_lock()
            lock_active = self._lock_is_active(lock)
            lock_stale = lock_active and self._active_lock_is_stale(lock)

            if not lock_active or lock_stale:
                if self._try_acquire():
                    return
            if time.time() >= deadline:
                raise LockAcquireTimeoutError(
                    f"Не удалось захватить lease-lock '{self._lock_key}' "
                    f"за {self._wait_timeout_sec} сек (S3-бэкенд task tracker'а)"
                )
            time.sleep(self._poll_interval_sec)

    def _try_acquire(self) -> bool:
        now = time.time()
        candidate = {
            "owner_id": self._owner_id,
            "acquired_at": now,
            "expires_at": now + self._lease_seconds,
        }
        self._write_json(self._lock_key, candidate)

        confirmed = self._read_lock()
        return confirmed is not None and confirmed.get("owner_id") == self._owner_id

    def _release_lock(self) -> None:
        released = {
            "owner_id": self._owner_id,
            "acquired_at": time.time(),
            "expires_at": 0.0,
        }
        self._write_json(self._lock_key, released)

    def _apply_retention(self, state: dict[str, TaskState]) -> dict[str, TaskState]:
        cutoff = time.time() - self._retention_months * _SECONDS_PER_MONTH
        return {
            task_id: task
            for task_id, task in state.items()
            if task.started_at >= cutoff
        }

    @staticmethod
    def _serialize_state(state: dict[str, TaskState]) -> dict[str, Any]:
        tasks_raw = {}
        for task_id, task in state.items():
            raw = asdict(task)
            raw["status"] = task.status.value
            tasks_raw[task_id] = raw
        return {"tasks": tasks_raw}

    @staticmethod
    def _deserialize_state(raw: dict[str, Any]) -> dict[str, TaskState]:
        tasks_raw = raw.get("tasks", {})
        result: dict[str, TaskState] = {}
        for task_id, fields in tasks_raw.items():
            fields = dict(fields)
            fields["status"] = TaskStatus(fields["status"])
            result[task_id] = TaskState(**fields)
        return result

    def _read_json(self, key: str) -> dict[str, Any] | None:
        try:
            with self._fs.open(key, "rb") as f:
                content = f.read()
        except FileNotFoundError:
            return None
        if not content:
            return None
        return json.loads(content)

    def _write_json(self, key: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        with self._fs.open(key, "wb") as f:
            f.write(body)

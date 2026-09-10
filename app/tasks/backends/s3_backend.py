"""
Best-effort координация через S3 lease: два постоянных объекта, без DELETE.

Сервер не поддерживает условные записи. Подтверждение с задержкой и проверки
владельца снижают вероятность гонки, но конкурентный PUT всё ещё может пройти
между проверкой и записью. Это не атомарный распределённый lock. Все процессы
записи должны использовать этот backend и синхронизированные часы; перехват
чужого lock допускается только после истечения lease.
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any

from app.tasks.backends.base import MutateFn, TaskStorageBackend
from app.tasks.state import TaskState, TaskStatus

logger = logging.getLogger(__name__)

DEFAULT_LEASE_SECONDS = 60
DEFAULT_WAIT_TIMEOUT_SECONDS = 300
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
LOCK_CONFIRMATION_DELAY_SECONDS = 2.0
LOCK_WRITE_MARGIN_SECONDS = 5.0
DEFAULT_RETENTION_MONTHS = 6

_SECONDS_PER_MONTH = 30 * 24 * 60 * 60
_TERMINAL_STATUSES = (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.ABORTED)


class LockAcquireTimeoutError(Exception):
    """Не удалось захватить локальный lock или S3 lease за отведённое время."""


class LockLostError(Exception):
    """Владение потеряно, lease истёк или оставшегося времени мало для записи."""


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
        for name, value in (
            ("lease_seconds", lease_seconds),
            ("wait_timeout_sec", wait_timeout_sec),
            ("poll_interval_sec", poll_interval_sec),
            ("retention_months", retention_months),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if lease_seconds <= LOCK_CONFIRMATION_DELAY_SECONDS + LOCK_WRITE_MARGIN_SECONDS:
            raise ValueError("lease_seconds must exceed confirmation delay plus write margin")
        if not state_key or not lock_key or state_key == lock_key:
            raise ValueError("state_key and lock_key must be distinct nonempty keys")

        self._fs = s3_fs
        self._state_key = state_key
        self._lock_key = lock_key
        self._lease_seconds = lease_seconds
        self._wait_timeout_sec = wait_timeout_sec
        self._poll_interval_sec = poll_interval_sec
        self._retention_months = retention_months
        self._owner_id = uuid.uuid4().hex
        self._local_lock = threading.Lock()

    def initialize(self) -> None:
        """Создать два постоянных объекта при старте, сохранив существующие задачи."""
        if self._read_json(self._state_key) is None or self._read_lock() is None:
            self.mutate(lambda current: current)

    def read_state(self) -> dict[str, TaskState]:
        raw = self._read_json(self._state_key)
        return {} if raw is None else self._deserialize_state(raw)

    def mutate(self, fn: MutateFn) -> dict[str, TaskState]:
        # Общий deadline включает ожидание lock, занятого другим потоком процесса.
        deadline = time.monotonic() + self._wait_timeout_sec
        if not self._local_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise self._timeout_error()
        token = None
        try:
            token = self._wait_and_acquire_lock(deadline)
            state = self.read_state()
            self._assert_ownership(token)
            new_state = self._apply_retention(fn(state))
            # Сериализация может затянуться или упасть — выполняем её до последней проверки.
            body = self._encode_json(self._serialize_state(new_state))
            self._assert_ownership(token)
            self._write_bytes(self._state_key, body)
            return new_state
        finally:
            if token is not None:
                self._release_lock(token)
            self._local_lock.release()

    def cleanup(self) -> bool:
        """Удалить устаревшие записи перезаписью состояния, только если они есть.

        Предварительная проверка не захватывает и не записывает lock. Retention
        проверяется повторно внутри mutate, чтобы сохранить недавно обновлённые
        задачи.
        """
        state = self.read_state()
        if len(self._apply_retention(state)) == len(state):
            return False
        self.mutate(lambda current: current)
        return True

    def _read_lock(self) -> dict[str, Any] | None:
        return self._read_json(self._lock_key)

    def _lock_is_active(self, lock: dict[str, Any] | None) -> bool:
        return lock is not None and time.time() < lock.get("expires_at", 0.0)

    def _owns_lock(
        self, lock: dict[str, Any] | None, token: str, remaining_seconds: float = 0.0
    ) -> bool:
        return (
            lock is not None
            and lock.get("owner_id") == self._owner_id
            and lock.get("token") == token
            and time.time() + remaining_seconds < lock.get("expires_at", 0.0)
        )

    def _assert_ownership(self, token: str) -> None:
        if not self._owns_lock(self._read_lock(), token, LOCK_WRITE_MARGIN_SECONDS):
            raise LockLostError(f"S3 lease '{self._lock_key}' was lost or is about to expire")

    def _timeout_error(self) -> LockAcquireTimeoutError:
        return LockAcquireTimeoutError(
            f"Could not acquire S3 lease '{self._lock_key}' within {self._wait_timeout_sec}s"
        )

    def _wait_and_acquire_lock(self, deadline: float | None = None) -> str:
        if deadline is None:
            deadline = time.monotonic() + self._wait_timeout_sec
        while time.monotonic() < deadline:
            lock = self._read_lock()
            if not self._lock_is_active(lock) and time.monotonic() < deadline:
                token = self._try_acquire(deadline)
                if token is not None:
                    return token
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Случайная задержка не даёт подам повторять попытки захвата синхронно.
            delay = random.uniform(0.5 * self._poll_interval_sec, 1.5 * self._poll_interval_sec)
            time.sleep(min(delay, remaining))
        raise self._timeout_error()

    def _try_acquire(self, deadline: float | None = None) -> str | None:
        if deadline is None:
            deadline = time.monotonic() + self._wait_timeout_sec
        if deadline - time.monotonic() < LOCK_CONFIRMATION_DELAY_SECONDS:
            return None
        token = uuid.uuid4().hex
        now = time.time()
        candidate = {
            "owner_id": self._owner_id,
            "token": token,
            "acquired_at": now,
            "expires_at": now + self._lease_seconds,
        }
        acquired = False
        try:
            self._write_json(self._lock_key, candidate)
            if not self._owns_lock(self._read_lock(), token):
                return None
            if deadline - time.monotonic() < LOCK_CONFIRMATION_DELAY_SECONDS:
                return None
            time.sleep(LOCK_CONFIRMATION_DELAY_SECONDS)
            confirmed = self._read_lock()
            acquired = (
                self._owns_lock(confirmed, token, LOCK_WRITE_MARGIN_SECONDS)
                and time.monotonic() < deadline
            )
            return token if acquired else None
        finally:
            if not acquired:
                self._release_lock(token)

    def _release_lock(self, token: str) -> None:
        # Проверка и освобождение на этом S3-сервере всё ещё не атомарны.
        # Не перезаписываем lock другой попытки, даже если она из нашего процесса.
        try:
            current = self._read_lock()
            if not self._owns_lock(current, token):
                return
            released = dict(current)
            released["expires_at"] = 0.0
            self._write_json(self._lock_key, released)
        except Exception:
            # После истечения lease lock снова станет доступен. Ошибка unlock
            # не должна скрывать успешную запись или исходную ошибку мутации/S3.
            logger.warning(
                "Could not release S3 lease '%s'; waiting for expiry", self._lock_key, exc_info=True
            )

    def _apply_retention(self, state: dict[str, TaskState]) -> dict[str, TaskState]:
        cutoff = time.time() - self._retention_months * _SECONDS_PER_MONTH
        retained = {}
        for task_id, task in state.items():
            if task.status in _TERMINAL_STATUSES:
                last_seen = task.finished_at if task.finished_at is not None else task.updated_at
            else:
                last_seen = task.heartbeat_at if task.heartbeat_at is not None else task.updated_at
            if last_seen >= cutoff:
                retained[task_id] = task
        return retained

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
        result: dict[str, TaskState] = {}
        for task_id, fields in raw.get("tasks", {}).items():
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
        return json.loads(content) if content else None

    @staticmethod
    def _encode_json(payload: dict[str, Any]) -> bytes:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _write_json(self, key: str, payload: dict[str, Any]) -> None:
        self._write_bytes(key, self._encode_json(payload))

    def _write_bytes(self, key: str, body: bytes) -> None:
        with self._fs.open(key, "wb") as f:
            f.write(body)

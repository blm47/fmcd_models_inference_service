"""
Контракт физического хранения. Callback содержит только изменения состояния в памяти.
"""

from collections.abc import Callable
from typing import Protocol, TypeVar

from app.tasks.types import TaskState

T = TypeVar("T")


class TaskStorageError(RuntimeError):
    """
    Хранилище очереди недоступно или операция не подтверждена.
    """


class TaskBackend(Protocol):
    supports_retention: bool

    def initialize(self) -> None: ...

    def get(self, task_id: str) -> TaskState | None: ...

    def get_by_key(self, key: str) -> TaskState | None: ...

    def list_active(self) -> list[TaskState]: ...

    def mutate(
        self,
        change: Callable[[dict[str, TaskState]], T],
        *,
        task_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> T:
        """
        Атомарно применяет callback и возвращает результат после подтверждения записи.

        Снимок содержит все незавершённые задачи и запись по переданному ID/ключу,
        даже если она завершена. Backend может передать дополнительные записи.
        Callback идемпотентен: backend вправе повторить его после конфликта записи.
        Внутри callback запрещены I/O и побочные эффекты вне переданного снимка.
        Удаление истории выполняется через cleanup, если supports_retention=True.
        """
        ...

    def cleanup(self, cutoff: float) -> None: ...

    def close(self) -> None: ...

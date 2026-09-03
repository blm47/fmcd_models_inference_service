"""
Абстрактный контракт физического хранения состояния задач.

TaskManager/TaskStore обращаются к состоянию задач ТОЛЬКО через этот
интерфейс. Ни TaskManager, ни роуты не должны знать, что физически стоит
за ним - S3 с lease-lock файлом, Redis, Postgres. Смена движка хранения =
смена backend-класса в lifespan (app/main.py), без изменений в
TaskManager/worker/routes.

Разделение операций:
  - read_state()  - "грязное" чтение без захвата critical section.
    Используется только для GET /status (не мутирует состояние), поэтому
    не должно требовать lock.
  - mutate(fn)     - критическая секция read-modify-write. fn получает
    текущий dict[task_id -> TaskState] и возвращает обновлённый dict,
    который атомарно (в рамках гарантий backend'а) записывается обратно.
    Все операции, изменяющие состояние (add/update_progress/set_status),
    идут через mutate(), чтобы backend мог обернуть их в лок ровно один раз.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.tasks.state import TaskState

MutateFn = Callable[[dict[str, "TaskState"]], dict[str, "TaskState"]]


class TaskStorageBackend(abc.ABC):
    """
    Абстрактный backend физического хранения состояния задач.

    Реализации: S3LeaseLockBackend (сейчас), в будущем - RedisBackend,
    PostgresBackend. Контракт минимальный и намеренно НЕ включает понятия
    "lock"/"lease"/"файл" - это деталь конкретного backend'а.
    """

    @abc.abstractmethod
    def read_state(self) -> dict[str, "TaskState"]:
        """
        Читает текущее состояние без захвата critical section.
        Может вернуть чуть устаревшие данные при гонке с параллельной
        записью - это осознанный трейд-офф для read-only статусных
        запросов (GET /tasks/{id}/status), которые не должны блокироваться
        на lock ради read-only операции.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mutate(self, fn: MutateFn) -> dict[str, "TaskState"]:
        """
        Атомарно (относительно других mutate() вызовов, в т.ч. из других
        подов) выполняет: read -> fn(state) -> write. Возвращает итоговое
        записанное состояние.

        Реализация backend'а сама решает, как обеспечить взаимное
        исключение (lease-lock файл в S3, Redis-лок, Postgres FOR UPDATE),
        но снаружи это выглядит как одна атомарная операция.
        """
        raise NotImplementedError

"""
Cooperative cancellation: worker.py сам проверяет флаг между чанками и
завершает работу, а не убивается снаружи (что небезопасно для GPU-памяти
и файловой записи в S3 — можно оставить недописанный output.parquet).
"""

import threading


class CancellationRegistry:
    """dict[task_id -> threading.Event], потокобезопасный."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}

    def create(self, task_id: str) -> threading.Event:
        event = threading.Event()
        with self._lock:
            self._events[task_id] = event
        return event

    def request_cancel(self, task_id: str) -> bool:
        with self._lock:
            event = self._events.get(task_id)
        if event is None:
            return False
        event.set()
        return True

    def is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            event = self._events.get(task_id)
        return event.is_set() if event else False

    def cleanup(self, task_id: str) -> None:
        with self._lock:
            self._events.pop(task_id, None)

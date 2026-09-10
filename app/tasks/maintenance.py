"""Периодическая очистка истории задач, в том числе в простое без инференса."""

import logging
import random
import threading

from app.tasks.state import TaskStore


class TaskMaintenance:
    def __init__(
            self, 
            store: TaskStore, 
            interval_sec: float, 
            logger: logging.Logger = logging.getLogger(__name__)
        ):
        self._store = store
        self._interval = interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="task-history-cleanup", daemon=True)
        self._logger = logger

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # Разносим запуск очистки, чтобы стартующие вместе поды не спорили за lease.
        delay = random.uniform(0.0, min(30.0, self._interval))
        while not self._stop.wait(delay):
            try:
                self._store.cleanup()
            except Exception:
                self._logger.exception("Task history cleanup failed; next pass will retry")
            delay = self._interval * random.uniform(0.9, 1.1)

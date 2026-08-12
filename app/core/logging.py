"""
Настройка структурного логирования.

Почему отдельный модуль: логи из worker.py (фоновая задача) и routes (запрос/ответ)
должны быть тегированы task_id, чтобы можно было грепать логи одной задачи
отдельно от остальных при диагностике долгого 2-3М-строчного инференса.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | task_id=%(task_id)s | %(message)s",
        defaults={"task_id": "-"},
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def get_task_logger(logger_name: str, task_id: str) -> logging.LoggerAdapter:
    """Возвращает logger с уже прибитым task_id в каждой строке."""
    base = logging.getLogger(logger_name)
    return logging.LoggerAdapter(base, extra={"task_id": task_id})

"""Единственная точка получения кастомного logger из dadm_functions."""

from typing import Any


def setup_logging() -> Any:
    """Возвращает общий объект библиотеки с методами info, warn и error."""
    from dadm_functions.logger import logger

    return logger

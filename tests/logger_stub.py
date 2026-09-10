"""Строгий контракт кастомного logger: только три метода и один аргумент."""

from unittest.mock import create_autospec


class CustomLogger:
    def info(self, message: str):
        pass

    def warn(self, message: str):
        pass

    def error(self, message: str):
        pass


def make_logger():
    return create_autospec(CustomLogger, instance=True, spec_set=True)

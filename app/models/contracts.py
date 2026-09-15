"""
Лёгкие контракты pipeline без импорта библиотек моделей и чтения артефактов.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def column_names(value: object, field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    """
    Проверяет список имён, сохраняя порядок колонок из контракта.
    """
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise ValueError(f"{field_name}: ожидается список имён колонок")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"{field_name}: имена колонок должны быть непустыми строками")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name}: имена колонок должны быть уникальными")
    return tuple(value)


@dataclass(frozen=True)
class ModelSpec:
    """
    Настройки одной модели из YAML; создание не обращается к её артефактам.
    """

    name: str
    backend: str
    artifacts_dir: Path
    id_cols: tuple[str, ...]
    device: str
    infer_batch_size: int
    parquet_read_chunk_size: int
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("name", "backend", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}: ожидается непустая строка")
        if not isinstance(self.artifacts_dir, (str, Path)) or not str(self.artifacts_dir).strip():
            raise ValueError("artifacts_dir: ожидается путь к каталогу артефактов")
        object.__setattr__(self, "artifacts_dir", Path(self.artifacts_dir))
        object.__setattr__(self, "id_cols", column_names(self.id_cols, "id_cols"))
        for name in ("infer_batch_size", "parquet_read_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}: ожидается положительное целое число")
        if not isinstance(self.options, Mapping):
            raise ValueError("options: ожидается словарь параметров backend")
        if any(not isinstance(key, str) or not key.strip() for key in self.options):
            raise ValueError("options: имена параметров должны быть непустыми строками")
        if set(self.options) & self.__dataclass_fields__.keys():
            raise ValueError("options не должен дублировать общие настройки модели")
        object.__setattr__(self, "options", dict(self.options))


class ModelBundle(ABC):
    """
    Pipeline одной задачи. Конструктор сохраняет только spec и общий logger.

    Подкласс освобождает ресурсы в close, в том числе после частичной ошибки load.
    Повторный close безопасен. Worker вызывает close в finally до следующей задачи.
    """

    def __init__(self, spec: ModelSpec, logger: Any) -> None:
        self.spec = spec
        self.logger = logger
        self.required_columns: tuple[str, ...] = ()
        self.output_columns: tuple[str, ...] = ()

    @abstractmethod
    def load(self) -> None:
        """
        Загружает pipeline и задаёт required_columns/output_columns без ключей.
        """

    @abstractmethod
    def predict_batch(
        self, frame: pd.DataFrame, check_shutdown: Callable[[], None]
    ) -> pd.DataFrame:
        """
        Возвращает одну строку на входную, сохраняя порядок и значения id_cols.

        Вход не изменяется; схема выхода стабильна и соответствует output_columns.
        Проверка остановки вызывается между дорогими стадиями pipeline.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Идемпотентно освобождает ресурсы, включая частично загруженный pipeline.
        """

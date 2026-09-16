"""
Точка подключения API model registry для скачивания файлов на локальный диск пода.
"""

from pathlib import Path
from typing import Any


def download_model_artifacts(name: str, artifacts_dir: str | Path, logger: Any) -> None:
    """
    Заглушка скачивания модели и всех артефактов в artifacts_dir при старте пода.

    Пока использует заранее размещённые локальные файлы. Не читает схемы,
    не загружает веса в память и не проверяет комплектность. MODEL-002 заменит
    тело метода обращением к API registry; ошибка скачивания должна прервать startup.
    """
    logger.info(
        f"Модель '{name}': заглушка скачивания из model registry; "
        f"используются локальные артефакты в {artifacts_dir}"
    )

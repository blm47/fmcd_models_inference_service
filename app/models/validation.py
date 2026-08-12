"""
Валидация входного parquet перед постановкой задачи в фон.

Требование из архитектуры: ручка /infer должна СИНХРОННО проверить, что во
входном parquet присутствуют все фичи (NUM_COLS + CAT_COLS), необходимые
модели, и уйти в фон только после успешной проверки. Технические колонки
(ID_COL, DATE_COL) также обязательны, т.к. без них нельзя собрать результат.
"""

from dataclasses import dataclass

import pyarrow.parquet as pq

from app.models.registry import ModelBundle

ID_COL = "customer_mdm_id"
DATE_COL = "partition_report_dt"


@dataclass
class ValidationResult:
    is_valid: bool
    missing_columns: list[str]
    total_rows: int


def validate_input_parquet(s3_path: str, bundle: ModelBundle, s3_client) -> ValidationResult:
    """
    Читает только метаданные/схему parquet-файла из S3 (без загрузки данных в память),
    чтобы проверка была дешёвой даже для файлов на 2-3 млн строк.
    """
    parquet_file = s3_client.open_parquet_file(s3_path)
    available_columns = set(parquet_file.schema.names)

    required_columns = set(bundle.num_cols) | set(bundle.cat_cols) | {ID_COL, DATE_COL}
    missing = sorted(required_columns - available_columns)

    return ValidationResult(
        is_valid=len(missing) == 0,
        missing_columns=missing,
        total_rows=parquet_file.metadata.num_rows,
    )

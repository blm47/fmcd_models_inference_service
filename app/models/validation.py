"""
Валидация входного parquet перед постановкой задачи в фон.

ручка /infer должна проверить, что во
входном parquet присутствуют все фичи, необходимые
моделям, и уйти в фон только после успешной проверки. Технические колонки
(ID_COL, DATE_COL) также обязательны, т.к. без них нельзя собрать результат.
"""

from dataclasses import dataclass

from app.models.registry import ModelBundle
from app.storage.s3_client import S3Client

# ID_COL = "customer_mdm_id"
# DATE_COL = "partition_report_dt"


@dataclass
class ValidationResult:
    is_valid: bool
    missing_columns: list[str]
    total_rows: int


def validate_input_parquet(s3_input_prefix: str, bundle: ModelBundle, s3_client: S3Client) -> ValidationResult:
    """
    Открывает входной префикс как pyarrow.dataset (объединяет схему всех
    парт-файлов под префиксом без чтения данных построчно), затем:
      1. Сверяет набор колонок со списком требуемых фич модели.
      2. Считает total_rows через метаданные всех парт-файлов (дёшево,
         не грузит сами данные в память) - нужно для total_rows в TaskState
         и последующего расчёта progress_pct/eta_seconds.
    """

    available_columns = s3_client.get_schema_columns(s3_input_prefix)

    required_columns = set(bundle.num_cols) | set(bundle.cat_cols) | set(bundle.id_cols)
    missing = sorted(required_columns - available_columns)

    total_rows = s3_client.count_rows(s3_input_prefix) if not missing else 0

    return ValidationResult(
        is_valid=len(missing) == 0,
        missing_columns=missing,
        total_rows=total_rows,
    )

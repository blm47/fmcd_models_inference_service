"""Экспорт текущего контракта FastAPI в читаемый файл OpenAPI."""

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "openapi.yaml"
# Скрипт запускается из scripts/, поэтому явно добавляем корень проекта для импорта app.
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402


def main() -> None:
    schema = app.openapi()
    OUTPUT_PATH.write_text(
        yaml.safe_dump(schema, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    print(f"OpenAPI сохранён в {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

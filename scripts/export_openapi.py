"""Экспорт текущего контракта FastAPI в читаемый файл OpenAPI."""

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "openapi.yaml"
MARKDOWN_PATH = PROJECT_ROOT / "docs" / "api.md"
# Скрипт запускается из scripts/, поэтому явно добавляем корень проекта для импорта app.
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402


def schema_type(schema: dict) -> str:
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    for union in ("anyOf", "oneOf", "allOf"):
        if union in schema:
            separator = " & " if union == "allOf" else " | "
            return separator.join(schema_type(item) for item in schema[union])
    if schema.get("type") == "array":
        return f"array<{schema_type(schema.get('items', {}))}>"
    return schema.get("type", "any")


def cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def render_markdown(schema: dict) -> str:
    """Формирует таблицы API и отдельную UML-карточку для каждого DTO."""
    lines = [
        "# API и DTO сервиса инференса",
        "",
        "Файл сформирован из FastAPI командой `python scripts/export_openapi.py`.",
        "Изменяйте маршруты и Pydantic-модели, затем повторяйте экспорт.",
        "",
        schema["info"].get("description", ""),
        "",
        "## Методы API",
        "",
        "| Метод | Назначение | Тело запроса | Ответы |",
        "| --- | --- | --- | --- |",
    ]
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete", "head", "options", "trace"}:
                continue
            request = operation.get("requestBody", {}).get("content", {})
            request_type = schema_type(request.get("application/json", {}).get("schema", {}))
            responses = []
            for code, response in operation["responses"].items():
                body = response.get("content", {}).get("application/json", {}).get("schema")
                label = schema_type(body) if body else "без тела"
                responses.append(f"`{code}` {cell(label)}")
            lines.append(
                f"| `{method.upper()} {path}` | {cell(operation.get('summary', ''))} | "
                f"{cell(request_type) if request else '—'} | {'<br>'.join(responses)} |"
            )
    lines.extend(
        [
            "",
            "## Структуры данных",
            "",
            "В UML показаны поля и типы. `null` означает допустимое пустое значение,",
            "а обязательность присутствия поля указана отдельно в таблице.",
            "Диаграммы отображаются в просмотрщике Markdown с поддержкой Mermaid.",
            "",
        ]
    )
    models = schema.get("components", {}).get("schemas", {})
    for name in models:
        lines.append(f"- [{name}](#{name.lower()})")
    for name, model in models.items():
        lines.extend(["", f"### {name}", ""])
        if model.get("description"):
            lines.extend([model["description"], ""])
        lines.extend(["```mermaid", "classDiagram", f"    class {name} {{"])
        if "enum" in model:
            lines.append("        <<enumeration>>")
            lines.extend(f"        {value}" for value in model["enum"])
        for field, definition in model.get("properties", {}).items():
            # Mermaid использует тильды для обобщённых типов и не принимает union через |.
            type_name = schema_type(definition).replace(" | ", " or ").replace(" & ", " and ")
            type_name = type_name.replace("<", "~").replace(">", "~")
            lines.append(f"        {type_name} {field}")
        lines.extend(["    }", "```", ""])
        if "enum" in model:
            lines.extend(["Значения: " + ", ".join(f"`{v}`" for v in model["enum"]) + ".", ""])
        if not model.get("properties"):
            continue
        lines.extend(
            [
                "| Поле | Тип | Обязательно | Описание | Примеры и ограничения |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for field, definition in model["properties"].items():
            details = []
            for key in (
                "examples",
                "default",
                "enum",
                "minLength",
                "maxLength",
                "minimum",
                "maximum",
                "pattern",
                "format",
            ):
                if key in definition:
                    details.append(f"{key}: {json.dumps(definition[key], ensure_ascii=False)}")
            required = "да" if field in model.get("required", []) else "нет"
            lines.append(
                f"| `{field}` | `{cell(schema_type(definition))}` | {required} | "
                f"{cell(definition.get('description', '—'))} | {cell('; '.join(details)) or '—'} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    schema = app.openapi()
    OUTPUT_PATH.write_text(
        yaml.safe_dump(schema, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    MARKDOWN_PATH.write_text(render_markdown(schema), encoding="utf-8", newline="\n")
    print(f"OpenAPI сохранён в {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

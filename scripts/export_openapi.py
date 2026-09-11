"""Генерирует Markdown-снимок API для просмотра в веб-интерфейсе Git."""

import json
import sys
from html import escape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Позволяет запускать скрипт без установки проекта как пакета.
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402


def type_name(schema: dict) -> str:
    if "$ref" in schema:
        return schema["$ref"].rsplit("/", 1)[-1]
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            separator = " & " if key == "allOf" else " | "
            return separator.join(type_name(item) for item in schema[key])
    if schema.get("type") == "array":
        return f"array<{type_name(schema.get('items', {}))}>"
    return schema.get("type", "any")


def cell(value: str) -> str:
    return escape(value).replace("|", "&#124;").replace("\n", "<br>")


def schema_link(schema: dict) -> str:
    name = type_name(schema)
    if "$ref" in schema:
        return f"[{name}](#{name.lower()})"
    return cell(name)


def render_markdown(schema: dict) -> str:
    """Сохраняет структуру Swagger: операции, параметры, ответы и Schemas."""
    models = schema.get("components", {}).get("schemas", {})
    lines = [
        f"# {schema['info']['title']}",
        "",
        "Снимок API из FastAPI. Обновление: `python scripts/export_openapi.py`.",
        "",
        "Раскройте метод или DTO для просмотра. Примеры запросов нужно адаптировать",
        "к своим бакетам и модели. Отправка запросов доступна в `/docs` сервиса.",
        "",
        schema["info"].get("description", ""),
        "",
        "## Методы",
        "",
    ]
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "patch", "delete", "head", "options", "trace"}:
                continue
            title = f"{method.upper()} {path} — {operation.get('summary', '')}"
            lines.extend(
                [
                    "<details>",
                    f"<summary>{escape(title)}</summary>",
                    "",
                    operation.get("description", ""),
                    "",
                    "**Parameters**",
                    "",
                ]
            )
            parameters = operations.get("parameters", []) + operation.get("parameters", [])
            if parameters:
                lines.extend(
                    [
                        "| Имя | Где | Тип | Обязательно | Описание |",
                        "| --- | --- | --- | --- | --- |",
                    ]
                )
                for param in parameters:
                    lines.append(
                        f"| {cell(param['name'])} | {param['in']} | "
                        f"{schema_link(param.get('schema', {}))} | "
                        f"{'да' if param.get('required') else 'нет'} | "
                        f"{cell(param.get('description', '—'))} |"
                    )
            else:
                lines.append("Нет параметров пути или query.")
            body = operation.get("requestBody", {})
            for media, content in body.get("content", {}).items():
                body_schema = content.get("schema", {})
                lines.extend(
                    [
                        "",
                        "**Request body**",
                        "",
                        f"`{media}` · {schema_link(body_schema)} · "
                        f"{'обязательное' if body.get('required') else 'необязательное'} тело",
                        "",
                    ]
                )
                model = models.get(type_name(body_schema), body_schema)
                properties = model.get("properties", {})
                # Используем только примеры из контракта, не выдумываем значения ответов.
                example = {k: v["examples"][0] for k, v in properties.items() if v.get("examples")}
                if example and set(model.get("required", [])) <= example.keys():
                    lines.extend(
                        [
                            "Пример запроса:",
                            "",
                            "```json",
                            json.dumps(example, ensure_ascii=False, indent=2),
                            "```",
                            "",
                        ]
                    )
            lines.extend(
                [
                    "",
                    "**Responses**",
                    "",
                    "| HTTP | Описание | Тело ответа |",
                    "| --- | --- | --- |",
                ]
            )
            for code, response in operation["responses"].items():
                bodies = [
                    f"{media}: {schema_link(content.get('schema', {}))}"
                    for media, content in response.get("content", {}).items()
                ]
                lines.append(
                    f"| {code} | {cell(response.get('description', ''))} | "
                    f"{'<br>'.join(bodies) or '—'} |"
                )
            lines.extend(["", "</details>", ""])
    lines.extend(
        [
            "## Schemas",
            "",
            "Обязательное поле должно присутствовать в JSON. Тип `null` отдельно",
            "указывает, что значение может быть пустым.",
            "",
        ]
    )
    for name, model in models.items():
        lines.extend(
            [f"### {name}", "", "<details>", f"<summary>{name} — раскрыть схему</summary>", ""]
        )
        if model.get("description"):
            lines.extend([model["description"], ""])
        if "enum" in model:
            lines.extend(["Значения: " + ", ".join(f"`{v}`" for v in model["enum"]), ""])
        if model.get("properties"):
            lines.extend(
                [
                    "| Поле | Тип | Обязательно | Описание | Ограничения и примеры |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
        for field, definition in model.get("properties", {}).items():
            details = [
                f"{k}: {json.dumps(v, ensure_ascii=False)}"
                for k, v in definition.items()
                if k
                in {
                    "examples",
                    "default",
                    "enum",
                    "format",
                    "pattern",
                    "minLength",
                    "maxLength",
                    "minimum",
                    "maximum",
                    "minItems",
                    "maxItems",
                }
            ]
            lines.append(
                f"| {cell(field)} | {schema_link(definition)} | "
                f"{'да' if field in model.get('required', []) else 'нет'} | "
                f"{cell(definition.get('description', '—'))} | {cell('; '.join(details)) or '—'} |"
            )
        lines.extend(["", "</details>", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    (PROJECT_ROOT / "docs" / "swagger.md").write_text(
        render_markdown(app.openapi()), encoding="utf-8", newline="\n"
    )
    print("Сохранён docs/swagger.md")


if __name__ == "__main__":
    main()

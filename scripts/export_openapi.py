"""Генерирует автономный Swagger UI и OpenAPI без запуска сервиса."""

import base64
import sys
from html import escape
from pathlib import Path

import yaml
from fastapi.openapi.docs import get_swagger_ui_html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = PROJECT_ROOT / "docs"
# Позволяет запускать скрипт из любой директории без установки проекта как пакета.
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402


def render_html(schema: dict) -> str:
    """Использует шаблон FastAPI и встраивает схему, CSS и JavaScript."""
    assets = DOCS_PATH / "swagger-assets"
    javascript = (assets / "swagger-ui-bundle.js").read_text(encoding="utf-8")
    css = (assets / "swagger-ui.css").read_text(encoding="utf-8")
    parameters = {
        **(app.swagger_ui_parameters or {}),
        "spec": schema,
        # Снимок предназначен для чтения без сервера и внешнего валидатора.
        "supportedSubmitMethods": [],
        "validatorUrl": None,
    }
    page = get_swagger_ui_html(
        openapi_url="",
        title=escape(f"{app.title} — Swagger UI (снимок)"),
        swagger_js_url="snapshot-bundle.js",
        swagger_css_url="snapshot-style.css",
        swagger_favicon_url="data:,",
        swagger_ui_parameters=parameters,
    ).body.decode("utf-8")
    # Data URL сохраняет JavaScript без интерпретации его строк HTML-парсером.
    javascript_url = "data:text/javascript;base64," + base64.b64encode(
        javascript.encode("utf-8")
    ).decode("ascii")
    page = page.replace(
        '<link type="text/css" rel="stylesheet" href="snapshot-style.css">',
        f"<style>{css}</style>",
    )
    page = page.replace("snapshot-bundle.js", javascript_url)
    return page.replace("<head>", '<head>\n    <meta charset="utf-8">')


def main() -> None:
    schema = app.openapi()
    page = render_html(schema)
    (DOCS_PATH / "swagger.html").write_text(page, encoding="utf-8", newline="\n")
    (DOCS_PATH / "openapi.yaml").write_text(
        yaml.safe_dump(schema, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
        newline="\n",
    )
    print("Сохранены docs/swagger.html и docs/openapi.yaml")


if __name__ == "__main__":
    main()

"""
Подготавливает начальную DDL; с --apply выполняет её в PostgreSQL.
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg
from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import PostgresConfig, required_env  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402


def render_migration(schema: str, table_name: str):
    source = (PROJECT_ROOT / "migrations/001_create_task_store.sql").read_text(encoding="utf-8")
    return sql.SQL(source).format(
        schema=sql.Identifier(schema), table=sql.Identifier(schema, table_name)
    )


def main():
    parser = argparse.ArgumentParser(description="Начальная миграция очереди PostgreSQL")
    parser.add_argument("--apply", action="store_true", help="Выполнить DDL вместо вывода")
    args = parser.parse_args()
    schema = required_env("POSTGRES_SCHEMA")
    table_name = required_env("POSTGRES_TABLE_NAME")
    statement = render_migration(schema, table_name)
    if not args.apply:
        print(statement.as_string())
        return
    config = PostgresConfig(
        url=required_env("POSTGRES_URL"),
        login=required_env("POSTGRES_LOGIN"),
        password=required_env("POSTGRES_PASSWORD"),
        schema=schema,
        table_name=table_name,
        sslmode=os.environ.get("POSTGRES_SSLMODE", "prefer"),
    )
    logger = setup_logging()
    try:
        with psycopg.connect(
            config.url,
            user=config.login,
            password=config.password,
            sslmode=config.sslmode,
            connect_timeout=10,
        ) as connection:
            connection.execute(statement)
    except psycopg.Error as exc:
        logger.error(f"Миграция очереди PostgreSQL не выполнена: {type(exc).__name__}")
        raise SystemExit(1) from None
    logger.info("Начальная миграция очереди PostgreSQL выполнена")


if __name__ == "__main__":
    main()

"""
Каждая задача хранится в строке таблицы PostgreSQL. История не удаляется сервисом.
"""

import math
from contextlib import contextmanager
from dataclasses import asdict, fields

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.tasks.backends.base import TaskStorageError
from app.tasks.types import TaskState, TaskStatus

COLUMNS = tuple(field.name for field in fields(TaskState))
ACTIVE_FILTER = sql.SQL("status NOT IN ('DONE', 'FAILED', 'ABORTED')")


class PostgresTaskBackend:
    """
    Хранит задачи в PostgreSQL и управляет пулом соединений.

    Перед выдачей каждое соединение проверяется запросом к БД. Операции выполняются
    в коротких транзакциях: при успехе выполняется commit, при ошибке — rollback,
    после чего соединение возвращается в пул. Изменения очереди сериализуются
    блокировкой таблицы; история задач не удаляется сервисом.
    """

    supports_retention = False

    def __init__(self, postgres, config, logger):
        """
        Сохраняет настройки и общий logger, создаёт закрытый пул без подключения к БД.
        """
        self.postgres = postgres
        self.config = config
        self.logger = logger
        self.table = sql.Identifier(postgres.schema, postgres.table_name)
        self.columns = sql.SQL(", ").join(map(sql.Identifier, COLUMNS))
        self._pool = ConnectionPool(
            conninfo=self.postgres.url,
            kwargs={
                "user": self.postgres.login,
                "password": self.postgres.password,
                "sslmode": self.postgres.sslmode,
                "connect_timeout": math.ceil(self.config.connect_timeout_sec),
                "options": (
                    f"-c statement_timeout={math.ceil(self.config.read_timeout_sec * 1000)} "
                    f"-c lock_timeout={math.ceil(self.config.cas_timeout_sec * 1000)} "
                ),
                "row_factory": dict_row,
            },
            min_size=1,
            max_size=2,
            timeout=self.config.connect_timeout_sec,
            check=ConnectionPool.check_connection,
            open=False,
        )

    @contextmanager
    def _connection(self):
        """
        Выдаёт проверенное соединение на транзакцию и скрывает детали ошибок БД.
        """
        try:
            with self._pool.connection() as connection:
                connection.isolation_level = psycopg.IsolationLevel.READ_COMMITTED
                yield connection
        except psycopg.Error as exc:
            # Не включаем URL, пароль или текст ответа сервера в ошибку API/logger.
            self.logger.error(f"Ошибка очереди PostgreSQL: {type(exc).__name__}")
            raise TaskStorageError("Операция очереди PostgreSQL не подтверждена") from None

    def initialize(self):
        """
        Открывает пул и проверяет доступность таблицы и колонок без выполнения DDL.

        При ошибке проверки закрывает пул и передаёт исключение вызывающему коду.
        Ошибки БД и таймаут получения соединения преобразуются в TaskStorageError.
        """
        # DDL выполняет миграция; сервису не нужны права CREATE.
        self._pool.open()
        try:
            with self._connection() as connection:
                connection.execute(
                    sql.SQL("SELECT {} FROM {} LIMIT 0").format(self.columns, self.table)
                )
        except Exception:
            self._pool.close()
            raise

    def _read(self, connection, condition, params=()):
        rows = connection.execute(
            sql.SQL("SELECT {} FROM {} WHERE {}").format(self.columns, self.table, condition),
            params,
        ).fetchall()
        return {
            row["task_id"]: TaskState(**{**row, "status": TaskStatus(row["status"])})
            for row in rows
        }

    def get(self, task_id):
        """
        Возвращает задачу по task_id или None, если запись отсутствует.
        """
        with self._connection() as connection:
            return self._read(connection, sql.SQL("task_id = %s"), (task_id,)).get(task_id)

    def get_by_key(self, key):
        """
        Возвращает задачу по ключу идемпотентности или None, если ключ не найден.
        """
        with self._connection() as connection:
            tasks = self._read(connection, sql.SQL("idempotency_key = %s"), (key,))
            return next(iter(tasks.values()), None)

    def list_active(self):
        """
        Возвращает все незавершённые задачи, включая ожидающие в очереди.
        """
        with self._connection() as connection:
            return list(self._read(connection, ACTIVE_FILTER).values())

    def mutate(self, change, *, task_id=None, idempotency_key=None):
        """
        Атомарно применяет callback change и возвращает результат после commit.

        Callback получает незавершённые задачи и запись по task_id или ключу
        идемпотентности, включая завершённую. Сохраняются только новые и изменённые
        записи. Удаление записей вызывает ValueError; исключение callback откатывает
        транзакцию. Ошибки БД и пула преобразуются в TaskStorageError.
        """
        with self._connection() as connection:
            # Короткая блокировка сериализует решения об очереди, включая пустую таблицу.
            # Это защищает общий лимит, пересекающиеся пути и claim между подами.
            # Обычный SELECT не блокируется; load/infer выполняются вне транзакции.
            connection.execute(
                sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(self.table)
            )
            tasks = self._read(
                connection,
                ACTIVE_FILTER + sql.SQL(" OR task_id = %s OR idempotency_key = %s"),
                (task_id, idempotency_key),
            )
            before = {key: asdict(task) for key, task in tasks.items()}
            result = change(tasks)
            if before.keys() - tasks.keys():
                raise ValueError("PG backend не поддерживает удаление истории через mutate")
            for key, task in tasks.items():
                record = asdict(task)
                if record == before.get(key):
                    continue
                if key in before:
                    columns = [name for name in COLUMNS if name != "task_id"]
                    assignments = sql.SQL(", ").join(
                        sql.SQL("{} = %s").format(sql.Identifier(name)) for name in columns
                    )
                    connection.execute(
                        sql.SQL("UPDATE {} SET {} WHERE task_id = %s").format(
                            self.table, assignments
                        ),
                        [record[name] for name in columns] + [key],
                    )
                else:
                    connection.execute(
                        sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                            self.table,
                            self.columns,
                            sql.SQL(", ").join(sql.Placeholder() for _ in COLUMNS),
                        ),
                        [record[name] for name in COLUMNS],
                    )
            # При потере ответа COMMIT вызывающая сторона повторяет идемпотентную операцию.
        return result

    def cleanup(self, cutoff):
        """
        Не удаляет историю: retention управляется вне сервиса, cutoff игнорируется.
        """
        pass

    def close(self):
        """
        Закрывает пул; выданные соединения закрываются после возврата в него.

        Повторный вызов безопасен. После закрытия backend нельзя использовать снова.
        """
        self._pool.close()

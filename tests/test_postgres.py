"""
Контракт PG backend и интеграционные проверки на отдельной тестовой схеме.
"""

import os
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from unittest.mock import MagicMock, Mock, patch

import psycopg
from fake_s3 import config, enqueue
from logger_stub import make_logger
from psycopg import sql
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

from app.core.config import PostgresConfig
from app.tasks.backends.base import TaskStorageError
from app.tasks.backends.factory import create_task_backend
from app.tasks.backends.postgres import PostgresTaskBackend
from app.tasks.state import QueueConflictError, TaskState, TaskStatus, TaskStore
from scripts.migrate_task_store import render_migration


def postgres_config():
    return PostgresConfig("postgresql://localhost/test", "worker", "secret", "fmcd", "tasks")


class PostgresTests(unittest.TestCase):
    def setUp(self):
        self.logger = make_logger()
        patcher = patch("app.tasks.backends.postgres.ConnectionPool")
        self.pool_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.pool_class.check_connection = ConnectionPool.check_connection
        self.pool = self.pool_class.return_value
        self.backend = PostgresTaskBackend(postgres_config(), config(backend="pg"), self.logger)
        self.connection = MagicMock()
        self.connection.execute.return_value.fetchall.return_value = []
        self.connection.__enter__.return_value = self.connection
        self.pool.connection.return_value = self.connection
        self.store = TaskStore(self.backend, self.backend.config, self.logger)

    def test_pool_checks_connections_and_limits_size(self):
        options = self.pool_class.call_args.kwargs
        self.assertEqual(options["min_size"], 1)
        self.assertEqual(options["max_size"], 2)
        self.assertIs(options["check"], ConnectionPool.check_connection)
        self.assertFalse(options["open"])
        self.assertEqual(options["timeout"], self.backend.config.connect_timeout_sec)
        self.pool.open.assert_not_called()

    def test_initialize_opens_pool_and_checks_table(self):
        self.backend.initialize()
        self.pool.open.assert_called_once_with()
        self.pool.connection.assert_called_once_with()
        self.assertIn("LIMIT 0", self.connection.execute.call_args.args[0].as_string())
        self.pool.close.assert_not_called()

    def test_failed_initialize_closes_pool(self):
        self.connection.execute.side_effect = psycopg.OperationalError("secret")
        with self.assertRaises(TaskStorageError):
            self.backend.initialize()
        self.pool.close.assert_called_once_with()

    def test_close_releases_pool(self):
        self.backend.close()
        self.pool.close.assert_called_once_with()

    def test_pool_errors_are_reported_without_credentials(self):
        for exception in (PoolTimeout, PoolClosed):
            with self.subTest(exception=exception):
                self.pool.connection.side_effect = exception("secret")
                with self.assertRaises(TaskStorageError) as error:
                    self.backend.get("task")
                self.assertNotIn("secret", str(error.exception))
                self.assertNotIn("secret", self.logger.error.call_args.args[0])

    def test_factory_pg_does_not_create_s3_queue_client(self):
        settings = Mock(task_store=config(backend="pg"), postgres=postgres_config())
        with patch("app.tasks.backends.factory.boto3.client") as s3:
            self.assertIsInstance(create_task_backend(settings, self.logger), PostgresTaskBackend)
        s3.assert_not_called()

    def test_no_retention_calls_or_connections_for_pg(self):
        self.store.cleanup()
        self.pool.connection.assert_not_called()

    def test_creation_is_committed_and_queue_lock_precedes_read(self):
        task = enqueue(self.store)
        calls = self.connection.execute.call_args_list
        self.assertTrue(calls[0].args[0].as_string().startswith('LOCK TABLE "fmcd"."tasks"'))
        self.assertIn("SHARE ROW EXCLUSIVE", calls[0].args[0].as_string())
        self.assertIn("status NOT IN", calls[1].args[0].as_string())
        self.assertEqual(calls[1].args[1], (None, "request-1"))
        self.assertTrue(calls[2].args[0].as_string().startswith("INSERT INTO"))
        self.assertIn(task.task_id, calls[2].args[1])
        self.connection.__exit__.assert_called_once_with(None, None, None)

    def test_callback_error_rolls_back_and_does_not_write(self):
        def fail(tasks):
            raise QueueConflictError("conflict")

        with self.assertRaises(QueueConflictError):
            self.backend.mutate(fail)
        self.assertEqual(self.connection.__exit__.call_args.args[0], QueueConflictError)
        self.assertEqual(self.connection.execute.call_count, 2)

    def test_update_loads_terminal_target_and_writes_only_changed_rows(self):
        task = TaskState(
            "task",
            "cc",
            "s3://in/data",
            "s3://out/data",
            TaskStatus.RUNNING,
            10,
            owner_id=self.store.owner_id,
        )
        self.connection.execute.return_value.fetchall.return_value = [asdict(task)]
        self.store.update_progress("task", 4, 2)
        calls = self.connection.execute.call_args_list
        self.assertEqual(calls[1].args[1], ("task", None))
        self.assertTrue(calls[2].args[0].as_string().startswith("UPDATE"))
        self.connection.reset_mock()
        task.status = TaskStatus.DONE
        self.connection.execute.return_value.fetchall.return_value = [asdict(task)]
        self.assertTrue(self.store.set_status("task", TaskStatus.DONE))
        self.assertEqual(self.connection.execute.call_count, 2)

    def test_failed_commit_is_reported_without_credentials(self):
        self.connection.__exit__.side_effect = psycopg.OperationalError("secret connection lost")
        with self.assertRaises(TaskStorageError) as error:
            enqueue(self.store)
        self.assertNotIn("secret", str(error.exception))
        self.assertNotIn("secret", self.logger.error.call_args.args[0])

    def test_migration_quotes_configured_identifiers(self):
        ddl = render_migration('Mixed"Schema', 'Task"Table').as_string()
        self.assertIn('CREATE TABLE "Mixed""Schema"."Task""Table"', ddl)
        self.assertIn("last_modified double precision NOT NULL", ddl)
        self.assertNotIn("heartbeat", ddl)


@unittest.skipUnless(os.environ.get("TEST_POSTGRES_URL"), "Не задан отдельный TEST_POSTGRES_URL")
class PostgresIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.schema = "fmcd_test_" + uuid.uuid4().hex
        self.pg = PostgresConfig(
            os.environ["TEST_POSTGRES_URL"],
            os.environ["TEST_POSTGRES_LOGIN"],
            os.environ["TEST_POSTGRES_PASSWORD"],
            self.schema,
            "tasks",
            os.environ.get("TEST_POSTGRES_SSLMODE", "prefer"),
        )
        self.addCleanup(self.drop_schema)
        with self.connection() as connection:
            connection.execute(render_migration(self.schema, "tasks"))
        self.first = self.store()
        self.second = self.store()

    def connection(self):
        return psycopg.connect(
            self.pg.url,
            user=self.pg.login,
            password=self.pg.password,
            sslmode=self.pg.sslmode,
            connect_timeout=5,
        )

    def store(self, **overrides):
        settings = config(backend="pg", **overrides)
        logger = make_logger()
        store = TaskStore(PostgresTaskBackend(self.pg, settings, logger), settings, logger)
        self.addCleanup(store.close)
        store.initialize()
        return store

    def drop_schema(self):
        # Удаляется только случайная схема, созданная этим тестом.
        with self.connection() as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(self.schema))
            )

    def concurrent(self, left, right):
        barrier = threading.Barrier(2)

        def run(action):
            barrier.wait(timeout=5)
            return action()

        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(run, action) for action in (left, right)]
            return [future.result(timeout=15) for future in futures]

    def test_duplicate_enqueue_and_single_claim_are_atomic(self):
        tasks = self.concurrent(lambda: enqueue(self.first), lambda: enqueue(self.second))
        self.assertEqual(tasks[0].task_id, tasks[1].task_id)
        claimed = self.concurrent(
            lambda: self.first.claim_next("one", {"cc"}),
            lambda: self.second.claim_next("two", {"cc"}),
        )
        self.assertEqual(sum(task is not None for task in claimed), 1)

    def test_overlapping_output_and_global_limit_are_atomic(self):
        limited = self.store(max_pending_tasks=1)

        def submit(store, key, path):
            try:
                return enqueue(store, key, path)
            except QueueConflictError:
                return None

        tasks = self.concurrent(
            lambda: submit(limited, "a", "s3://out/a"),
            lambda: submit(self.store(max_pending_tasks=1), "b", "s3://out/b"),
        )
        self.assertEqual(sum(task is not None for task in tasks), 1)
        active = self.first.get_all_active()[0]
        with self.assertRaises(QueueConflictError):
            enqueue(self.second, "overlap", active.s3_output_path + "/child")

    def test_terminal_history_survives_cleanup_and_idempotent_retries(self):
        task = enqueue(self.first)
        self.first.claim_next("one", {"cc"})
        self.first.update_progress(task.task_id, 10, 1)
        self.assertTrue(self.first.prepare_completion(task.task_id, 10, 1))
        self.assertTrue(self.first.set_status(task.task_id, TaskStatus.DONE))
        self.assertTrue(self.first.set_status(task.task_id, TaskStatus.DONE))
        self.first.cleanup()
        self.assertEqual(enqueue(self.second).task_id, task.task_id)
        self.assertEqual(self.second.get_all_active(), [])
        with self.assertRaises(QueueConflictError):
            enqueue(self.second, output="s3://out/changed")

    def test_concurrent_overlapping_paths_have_one_winner(self):
        def submit(store, key, path):
            try:
                return enqueue(store, key, path)
            except QueueConflictError:
                return None

        results = self.concurrent(
            lambda: submit(self.first, "a", "s3://out/data"),
            lambda: submit(self.second, "b", "s3://out/data/child"),
        )
        self.assertEqual(sum(task is not None for task in results), 1)

    def test_nonoverlapping_tasks_are_claimed_by_different_pods(self):
        enqueue(self.first, "a")
        enqueue(self.first, "b")
        tasks = self.concurrent(
            lambda: self.first.claim_next("one", {"cc"}),
            lambda: self.second.claim_next("two", {"cc"}),
        )
        self.assertEqual(len({task.task_id for task in tasks}), 2)
        self.assertEqual(self.first.claim_next("one", {"cc"}).task_id, tasks[0].task_id)

    def test_timeout_and_abort_cannot_be_overwritten_by_old_worker(self):
        task = enqueue(self.first)
        self.first.claim_next("one", {"cc"})
        self.second.request_abort(task.task_id)
        with self.connection() as connection:
            connection.execute(
                sql.SQL("UPDATE {} SET last_modified = 1 WHERE task_id = %s").format(
                    sql.Identifier(self.schema, "tasks")
                ),
                (task.task_id,),
            )
        self.second.fail_stale()
        self.first.update_progress(task.task_id, 10, 1)
        self.assertFalse(self.first.set_status(task.task_id, TaskStatus.DONE))
        self.assertIn("task_timeout", self.second.get(task.task_id).error)

    def test_callback_exception_rolls_back(self):
        task = enqueue(self.first)

        def fail(tasks):
            tasks[task.task_id].status = TaskStatus.FAILED
            raise ValueError("rollback")

        with self.assertRaises(ValueError):
            self.first.backend.mutate(fail)
        self.assertEqual(self.first.get(task.task_id).status, TaskStatus.QUEUED)

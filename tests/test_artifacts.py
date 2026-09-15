"""
Проверки startup-заглушки model registry и результата валидации входа в очереди.
"""

import asyncio
import threading
import types
import unittest
from unittest.mock import patch

from fake_s3 import FakeS3, make_store
from logger_stub import make_logger
from test_config import ENVIRONMENT

from app.main import app, lifespan
from app.models.artifacts import download_model_artifacts
from app.models.contracts import ModelSpec
from app.tasks.state import TaskStatus


class ArtifactsTests(unittest.TestCase):
    def test_successful_startup_keeps_only_specs_without_importing_backend(self):
        async def start():
            async with lifespan(app):
                self.assertTrue(
                    all(isinstance(spec, ModelSpec) for spec in app.state.models.values())
                )
                self.assertEqual(
                    set(app.state.models), {"fmcd_credit_cards", "fmcd_debet_cards", "fmcd_invest"}
                )

        with (
            patch.dict("os.environ", ENVIRONMENT),
            patch("app.main.setup_logging", return_value=make_logger()),
            patch("app.main.boto3.client"),
            patch("app.main.TaskStore"),
            patch("app.main.consume_queue"),
            patch("app.main.monitor_queue"),
            patch("app.main.install_shutdown_handlers", return_value={}),
            patch("app.main.restore_shutdown_handlers"),
            patch("app.storage.s3_client.S3Client"),
            patch("app.main.download_model_artifacts") as download,
            patch(
                "app.models.registry.import_module", side_effect=AssertionError("backend import")
            ),
        ):
            asyncio.run(start())
        self.assertEqual(download.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in download.call_args_list],
            ["fmcd_credit_cards", "fmcd_debet_cards", "fmcd_invest"],
        )

    def test_stub_does_not_access_files(self):
        logger = make_logger()
        with patch("pathlib.Path.open", side_effect=AssertionError("чтение файлов")):
            download_model_artifacts("cc", "missing/cc", logger)
        self.assertIn("заглушка", logger.info.call_args.args[0])

    def test_startup_downloads_each_model_before_initializing_services(self):
        models = [
            types.SimpleNamespace(name="cc", artifacts_dir="artifacts/cc"),
            types.SimpleNamespace(name="dc", artifacts_dir="artifacts/dc"),
        ]
        logger = make_logger()
        downloads = []

        def download(name, directory, passed_logger):
            downloads.append((name, directory))
            self.assertIs(passed_logger, logger)
            self.assertNotEqual(threading.current_thread(), threading.main_thread())
            if name == "dc":
                raise RuntimeError("registry unavailable")

        async def start():
            async with lifespan(app):
                self.fail("Startup не должен завершиться после ошибки скачивания")

        with (
            patch("app.main.setup_logging", return_value=logger),
            patch("app.main.load_settings", return_value=types.SimpleNamespace(models=models)),
            patch("app.main.download_model_artifacts", side_effect=download),
            patch("app.main.boto3.client") as client,
            self.assertRaisesRegex(RuntimeError, "registry unavailable"),
        ):
            asyncio.run(start())
        self.assertEqual(downloads, [("cc", "artifacts/cc"), ("dc", "artifacts/dc")])
        client.assert_not_called()


class InputMetadataTests(unittest.TestCase):
    def setUp(self):
        self.store = make_store(FakeS3())
        self.store.initialize()
        self.task = self.store.enqueue("key", "cc", "s3://in/data", "s3://out/data", None)

    def test_unknown_count_roundtrip_and_owner_update(self):
        self.assertIsNone(self.store.get(self.task.task_id).total_rows)
        self.store.claim_next("pod", {"cc"})
        before = self.store.get(self.task.task_id)
        self.assertTrue(self.store.set_total_rows(self.task.task_id, 15))
        after = self.store.get(self.task.task_id)
        self.assertEqual(after.total_rows, 15)
        self.assertEqual(after.updated_at, before.updated_at)

    def test_terminal_task_cannot_receive_row_count(self):
        self.store.claim_next("pod", {"cc"})
        self.store.set_status(self.task.task_id, TaskStatus.FAILED)
        self.assertFalse(self.store.set_total_rows(self.task.task_id, 15))
        self.assertIsNone(self.store.get(self.task.task_id).total_rows)

    def test_another_owner_cannot_change_row_count(self):
        self.store.claim_next("pod", {"cc"})
        other = make_store(self.store.client)
        with self.assertRaises(PermissionError):
            other.set_total_rows(self.task.task_id, 15)

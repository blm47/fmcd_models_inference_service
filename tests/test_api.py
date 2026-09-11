"""Контракт Airflow проверяется без загрузки моделей и GPU."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

from fake_s3 import FakeS3, make_store
from fastapi.testclient import TestClient
from logger_stub import make_logger

from app.main import app
from app.tasks.state import TaskStatus


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.store = make_store(FakeS3())
        self.store.initialize()
        app.state.task_store = self.store
        app.state.logger = make_logger()
        app.state.settings = types.SimpleNamespace(
            s3=types.SimpleNamespace(bucket_in="input", bucket_out="output"),
            task_store=self.store.config,
        )
        app.state.models = {"cc": object()}
        app.state.s3_client = Mock()
        app.state.s3_client.prefix_has_results.return_value = False
        validation = types.ModuleType("app.models.validation")
        validation.validate_input_parquet = Mock(
            return_value=types.SimpleNamespace(is_valid=True, total_rows=10)
        )
        self.validation = validation.validate_input_parquet
        patcher = patch.dict(sys.modules, {"app.models.validation": validation})
        patcher.start()
        self.addCleanup(patcher.stop)
        # Без контекстного менеджера TestClient не запускает GPU lifespan.
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.request = {
            "idempotency_key": "dag/run/task/attempt-1",
            "model_name": "cc",
            "s3_input_path": "s3://input/data/",
            "s3_output_path": "s3://output/data/",
        }

    def submit(self, **updates):
        return self.client.post("/infer", json={**self.request, **updates})

    def test_busy_pod_still_accepts_other_requests(self):
        first = self.submit().json()
        self.store.claim_next("pod1", {"cc"})
        second = self.submit(idempotency_key="other", s3_output_path="s3://output/other")
        self.assertEqual(second.status_code, 202)
        self.assertEqual(second.json()["status"], "QUEUED")
        self.assertIsNone(second.json()["pod_id"])
        self.assertEqual(self.store.get(first["task_id"]).status, TaskStatus.RUNNING)

    def test_duplicate_after_output_started_returns_original(self):
        first = self.submit().json()
        self.store.claim_next("pod1", {"cc"})
        app.state.s3_client.prefix_has_results.return_value = True
        second = self.submit()
        self.assertEqual(second.status_code, 202)
        self.assertEqual(second.json()["task_id"], first["task_id"])
        self.assertEqual(self.validation.call_count, 1)

    def test_new_attempt_after_failure_gets_new_id(self):
        first = self.submit().json()
        self.store.claim_next("pod1", {"cc"})
        self.store.set_status(first["task_id"], TaskStatus.FAILED)
        retry = self.submit(idempotency_key="dag/run/task/attempt-2")
        self.assertEqual(retry.status_code, 202)
        self.assertNotEqual(retry.json()["task_id"], first["task_id"])

    def test_status_and_abort_from_other_pod(self):
        task_id = self.submit().json()["task_id"]
        app.state.task_store = make_store(self.store.client)
        response = self.client.get(f"/tasks/{task_id}/status")
        self.assertEqual(response.json()["status"], "QUEUED")
        self.assertIsNone(response.json()["started_at"])
        self.assertEqual(
            self.client.get("/tasks/active").json()["active_tasks"][0]["task_id"], task_id
        )
        self.assertEqual(self.client.post(f"/tasks/{task_id}/abort").json()["status"], "ABORTED")

    def test_conflicting_key_or_output_gets_409(self):
        self.submit()
        self.assertEqual(self.submit(s3_output_path="s3://output/other").status_code, 409)
        self.assertEqual(self.submit(idempotency_key="other").status_code, 409)

    def test_storage_failure_does_not_confirm_acceptance(self):
        self.store.client.fail_next = 403
        self.assertEqual(self.submit().status_code, 503)

    def test_protects_queue_and_bucket_root(self):
        for path in (
            "s3://output/",
            "s3://output/_system",
            "s3://output/_system/fmcd_models",
            "s3://output//data",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.submit(s3_output_path=path).status_code, 422)

    def test_output_success_marker_rejects_new_attempt(self):
        app.state.s3_client.prefix_has_results.return_value = True
        self.assertEqual(self.submit().status_code, 422)

    def test_idempotency_key_required(self):
        del self.request["idempotency_key"]
        self.assertEqual(self.submit().status_code, 422)

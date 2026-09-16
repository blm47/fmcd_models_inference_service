"""
Readiness обслуживает очередь, liveness не зависит от S3/PG.
"""

import json
import threading
import unittest
from unittest.mock import Mock, patch

from fake_s3 import FakeS3, enqueue, make_store
from fastapi.testclient import TestClient
from logger_stub import make_logger

from app.main import app
from app.tasks.backends.base import TaskStorageError
from app.tasks.state import TaskStatus


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.store = make_store(FakeS3())
        self.store.initialize()
        self.consumer = Mock()
        self.consumer.is_alive.return_value = True
        self.stop = threading.Event()
        self.state = dict(app.state._state)
        self.addCleanup(self.restore_state)
        app.state.consumer = self.consumer
        app.state.stop = self.stop
        app.state.task_store = self.store
        app.state.logger = make_logger()
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def restore_state(self):
        app.state._state.clear()
        app.state._state.update(self.state)

    def test_readiness_expires_task_while_consumer_is_busy(self):
        task = enqueue(self.store)
        self.store.claim_next("pod", {"cc"})
        raw = json.loads(self.store.backend.client.body)
        raw["tasks"][task.task_id]["last_modified"] = 1
        self.store.backend.client.body = json.dumps(raw).encode()
        response = self.client.get("/healthz/readiness")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.get(task.task_id).status, TaskStatus.FAILED)

    def test_unavailable_storage_only_fails_readiness(self):
        with patch.object(
            self.store, "check_queue", side_effect=TaskStorageError("PG unavailable")
        ) as check:
            self.assertEqual(self.client.get("/healthz/liveness").status_code, 200)
            check.assert_not_called()
            response = self.client.get("/healthz/readiness")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json(), {"status": "unavailable"})
            self.assertEqual(self.client.get("/health").status_code, 503)
            self.assertEqual(self.client.get("/healthz/liveness").status_code, 200)

    def test_stopped_consumer_or_shutdown_fails_both_probes(self):
        for shutdown in (False, True):
            self.consumer.is_alive.return_value = shutdown
            if shutdown:
                self.stop.set()
            with patch.object(self.store, "check_queue") as check:
                for path in ("/healthz/readiness", "/healthz/liveness"):
                    self.assertEqual(self.client.get(path).status_code, 503)
                check.assert_not_called()

    def test_overlapping_readiness_does_not_queue_storage_requests(self):
        with self.store._maintenance_lock, patch.object(self.store, "fail_stale") as check:
            self.assertEqual(self.client.get("/healthz/readiness").status_code, 503)
            self.assertEqual(self.client.get("/healthz/liveness").status_code, 200)
            check.assert_not_called()

    def test_cleanup_is_periodic_and_failure_can_be_retried(self):
        with patch.object(self.store, "cleanup") as cleanup:
            with patch("app.tasks.state.time.monotonic", return_value=100):
                self.assertTrue(self.store.check_queue())
                self.assertTrue(self.store.check_queue())
                cleanup.assert_called_once()
            deadline = 100 + self.store.config.cleanup_interval_sec
            with patch("app.tasks.state.time.monotonic", return_value=deadline):
                cleanup.side_effect = RuntimeError("storage unavailable")
                with self.assertRaises(RuntimeError):
                    self.store.check_queue()
                cleanup.side_effect = None
                self.assertTrue(self.store.check_queue())
                self.assertEqual(cleanup.call_count, 3)

    def test_pg_readiness_never_invokes_retention_backend(self):
        backend = Mock(supports_retention=False)
        self.store.backend = backend
        backend.mutate.return_value = []
        self.assertTrue(self.store.check_queue())
        backend.cleanup.assert_not_called()

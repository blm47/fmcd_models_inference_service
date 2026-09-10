"""Таймауты не возобновляют старые задачи и допускают новую попытку в тот же путь."""

import json
import signal
import threading
import unittest
from unittest.mock import Mock, patch

from fake_s3 import FakeS3, enqueue, make_store

from app.core.shutdown import install_shutdown_handlers, restore_shutdown_handlers
from app.tasks.state import TaskStatus
from app.tasks.worker import monitor_queue


class TimeoutTests(unittest.TestCase):
    def setUp(self):
        self.s3 = FakeS3()
        self.worker = make_store(self.s3)
        self.monitor = make_store(self.s3)
        self.worker.initialize()
        self.task = enqueue(self.worker)
        self.worker.claim_next("pod1", {"cc"})

    def age(self, field):
        raw = json.loads(self.s3.body)
        raw["tasks"][self.task.task_id][field] = 1
        self.s3.body = json.dumps(raw).encode()

    def test_heartbeat_timeout_releases_same_output_for_new_attempt(self):
        self.age("heartbeat_at")
        self.monitor.fail_stale()
        failed = self.monitor.get(self.task.task_id)
        self.assertEqual(failed.status, TaskStatus.FAILED)
        self.assertIn("heartbeat_timeout", failed.error)
        retry = enqueue(self.monitor, "attempt-2", self.task.s3_output_path)
        self.assertEqual(retry.s3_output_path, self.task.s3_output_path)
        self.assertNotEqual(retry.task_id, self.task.task_id)

    def test_progress_timeout_works_with_live_heartbeat(self):
        self.age("updated_at")
        self.monitor.fail_stale()
        self.assertIn("progress_timeout", self.worker.get(self.task.task_id).error)

    def test_late_heartbeat_cannot_revive_expired_task(self):
        self.age("heartbeat_at")
        self.assertFalse(self.worker.heartbeat(self.task.task_id))
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_late_progress_does_not_revive_expired_task(self):
        self.age("heartbeat_at")
        self.worker.update_progress(self.task.task_id, 9, 1)
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_late_completion_does_not_replace_timeout(self):
        self.worker.prepare_completion(self.task.task_id, 10, 1)
        self.age("heartbeat_at")
        self.monitor.fail_stale()
        self.assertFalse(self.worker.set_status(self.task.task_id, TaskStatus.DONE))
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)
        with self.assertRaisesRegex(RuntimeError, "task_not_active"):
            self.worker.cancellation_requested(self.task.task_id)

    def test_timeout_covers_aborting_and_finalizing(self):
        for state in (TaskStatus.ABORTING, TaskStatus.FINALIZING):
            with self.subTest(state=state):
                raw = json.loads(self.s3.body)
                raw["tasks"][self.task.task_id].update(status=state, heartbeat_at=1)
                self.s3.body = json.dumps(raw).encode()
                self.monitor.fail_stale()
                self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_queued_and_done_are_not_expired(self):
        self.worker.prepare_completion(self.task.task_id, 10, 1)
        self.worker.set_status(self.task.task_id, TaskStatus.DONE)
        waiting = enqueue(self.worker, "waiting")
        raw = json.loads(self.s3.body)
        for task in raw["tasks"].values():
            task.update(heartbeat_at=1, updated_at=1)
        self.s3.body = json.dumps(raw).encode()
        writes = self.s3.writes
        self.monitor.fail_stale()
        self.assertEqual(self.s3.writes, writes)
        self.assertEqual(self.worker.get(waiting.task_id).status, TaskStatus.QUEUED)

    def test_monitor_runs_without_free_gpu_worker(self):
        self.age("heartbeat_at")
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        monitor_queue(self.monitor, stop, Mock())
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)


class ShutdownSignalTests(unittest.TestCase):
    def test_signal_sets_stop_and_preserves_uvicorn_handler(self):
        stop = threading.Event()
        original = Mock()
        with (
            patch("app.core.shutdown.signal.getsignal", return_value=original),
            patch("app.core.shutdown.signal.signal") as install,
        ):
            previous = install_shutdown_handlers(stop, Mock())
            handler = install.call_args_list[0].args[1]
            handler(signal.SIGTERM, None)
            self.assertTrue(stop.is_set())
            original.assert_called_once_with(signal.SIGTERM, None)
            restore_shutdown_handlers(previous)
            self.assertEqual(install.call_args_list[-1].args[1], original)

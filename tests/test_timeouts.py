"""
Таймауты не возобновляют старые задачи и допускают новую попытку в тот же путь.
"""

import json
import signal
import threading
import unittest
from unittest.mock import Mock, patch

from fake_s3 import FakeS3, enqueue, make_store

from app.core.shutdown import install_shutdown_handlers, restore_shutdown_handlers
from app.tasks.state import TaskStatus


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

    def test_task_timeout_releases_same_output_for_new_attempt(self):
        self.age("last_modified")
        self.monitor.fail_stale()
        failed = self.monitor.get(self.task.task_id)
        self.assertEqual(failed.status, TaskStatus.FAILED)
        self.assertIn("task_timeout", failed.error)
        retry = enqueue(self.monitor, "attempt-2", self.task.s3_output_path)
        self.assertEqual(retry.s3_output_path, self.task.s3_output_path)
        self.assertNotEqual(retry.task_id, self.task.task_id)

    def test_late_progress_does_not_revive_expired_task(self):
        self.age("last_modified")
        self.worker.update_progress(self.task.task_id, 9, 1)
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_late_completion_does_not_replace_timeout(self):
        self.worker.prepare_completion(self.task.task_id, 10, 1)
        self.age("last_modified")
        self.monitor.fail_stale()
        self.assertFalse(self.worker.set_status(self.task.task_id, TaskStatus.DONE))
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)
        with self.assertRaisesRegex(RuntimeError, "task_not_active"):
            self.worker.cancellation_requested(self.task.task_id)

    def test_timeout_covers_aborting_and_finalizing(self):
        for state in (TaskStatus.ABORTING, TaskStatus.FINALIZING):
            with self.subTest(state=state):
                raw = json.loads(self.s3.body)
                raw["tasks"][self.task.task_id].update(status=state, last_modified=1)
                self.s3.body = json.dumps(raw).encode()
                self.monitor.fail_stale()
                self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_queued_and_done_are_not_expired(self):
        self.worker.prepare_completion(self.task.task_id, 10, 1)
        self.worker.set_status(self.task.task_id, TaskStatus.DONE)
        waiting = enqueue(self.worker, "waiting")
        raw = json.loads(self.s3.body)
        for task in raw["tasks"].values():
            task.update(last_modified=1)
        self.s3.body = json.dumps(raw).encode()
        writes = self.s3.writes
        self.monitor.fail_stale()
        self.assertEqual(self.s3.writes, writes)
        self.assertEqual(self.worker.get(waiting.task_id).status, TaskStatus.QUEUED)

    def test_read_and_abort_do_not_extend_executor_lifetime(self):
        self.age("last_modified")
        before = self.worker.get(self.task.task_id).last_modified
        self.monitor.get_all_active()
        self.monitor.request_abort(self.task.task_id)
        self.assertEqual(self.worker.get(self.task.task_id).last_modified, before)
        self.monitor.fail_stale()
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_owner_updates_last_modified_and_next_timeout_uses_it(self):
        with patch("app.tasks.state.time.time", return_value=self.task.created_at + 60):
            self.worker.set_total_rows(self.task.task_id, 10)
            saved = self.worker.get(self.task.task_id).last_modified
            self.assertEqual(saved, self.task.created_at + 60)
        with patch(
            "app.tasks.state.time.time",
            return_value=saved + self.worker.config.task_timeout_sec - 1,
        ):
            self.monitor.fail_stale()
            self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.RUNNING)
        with patch(
            "app.tasks.state.time.time", return_value=saved + self.worker.config.task_timeout_sec
        ):
            self.monitor.fail_stale()
            self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_late_input_validation_cannot_revive_expired_task(self):
        self.age("last_modified")
        self.assertFalse(self.worker.set_total_rows(self.task.task_id, 15))
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_heartbeat_extends_lifetime_without_changing_progress(self):
        before = self.worker.get(self.task.task_id)
        with patch("app.tasks.state.time.time", return_value=before.last_modified + 30):
            self.assertTrue(self.worker.heartbeat(self.task.task_id))
        after = self.worker.get(self.task.task_id)
        self.assertEqual(after.last_modified, before.last_modified + 30)
        self.assertEqual(after.processed_rows, before.processed_rows)

    def test_late_heartbeat_does_not_revive_expired_task(self):
        self.age("last_modified")
        self.assertFalse(self.worker.heartbeat(self.task.task_id))
        self.assertEqual(self.worker.get(self.task.task_id).status, TaskStatus.FAILED)

    def test_heartbeat_from_other_owner_is_rejected(self):
        with self.assertRaises(PermissionError):
            self.monitor.heartbeat(self.task.task_id)


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

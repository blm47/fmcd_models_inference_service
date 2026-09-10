"""Проверки конкуренции, повторов запросов и безопасных переходов состояния."""

import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from botocore.exceptions import ClientError
from fake_s3 import FakeS3, enqueue, make_store

from app.tasks.state import QueueConflictError, TaskStatus


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.s3 = FakeS3()
        self.first = make_store(self.s3)
        self.second = make_store(self.s3)
        self.first.initialize()

    def concurrent(self, left, right):
        self.s3.barrier = threading.Barrier(2)
        self.s3.readers = 0
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(left), pool.submit(right)]
            return [future.result(timeout=5) for future in futures]

    def test_concurrent_enqueue_preserves_both_requests(self):
        tasks = self.concurrent(lambda: enqueue(self.first, "a"), lambda: enqueue(self.second, "b"))
        self.assertEqual(
            {t.task_id for t in tasks}, {t.task_id for t in self.first.get_all_active()}
        )

    def test_only_one_pod_claims_single_task(self):
        task = enqueue(self.first)
        claimed = self.concurrent(
            lambda: self.first.claim_next("pod1", {"cc"}),
            lambda: self.second.claim_next("pod2", {"cc"}),
        )
        self.assertEqual(sum(t is not None for t in claimed), 1)
        self.assertEqual(self.first.get(task.task_id).status, TaskStatus.RUNNING)

    def test_two_pods_claim_different_tasks(self):
        enqueue(self.first, "a")
        enqueue(self.first, "b")
        claimed = self.concurrent(
            lambda: self.first.claim_next("pod1", {"cc"}),
            lambda: self.second.claim_next("pod2", {"cc"}),
        )
        self.assertEqual(len({t.task_id for t in claimed}), 2)

    def test_concurrent_duplicate_request_has_one_task(self):
        tasks = self.concurrent(lambda: enqueue(self.first), lambda: enqueue(self.second))
        self.assertEqual(tasks[0].task_id, tasks[1].task_id)
        self.assertEqual(len(self.first.get_all_active()), 1)

    def test_lost_enqueue_response_does_not_duplicate(self):
        self.s3.lose_next_response = True
        task = enqueue(self.first)
        self.assertEqual(enqueue(self.second).task_id, task.task_id)
        self.assertEqual(len(self.first.get_all_active()), 1)

    def test_lost_claim_response_restores_owner(self):
        task = enqueue(self.first)
        self.s3.lose_next_response = True
        claimed = self.first.claim_next("pod1", {"cc"})
        self.assertEqual(claimed.task_id, task.task_id)
        self.assertIsNone(self.second.claim_next("pod2", {"cc"}))
        self.assertEqual(self.first.claim_next("pod1", {"cc"}).task_id, task.task_id)

    def test_lost_response_at_deadline_is_recovered_on_next_claim(self):
        enqueue(self.first)
        self.s3.lose_next_response = True
        with (
            patch("app.tasks.state.time.monotonic", side_effect=[0, 31]),
            self.assertRaises(TimeoutError),
        ):
            self.first.claim_next("pod1", {"cc"})
        self.assertIsNotNone(self.first.claim_next("pod1", {"cc"}))
        self.assertIsNone(self.second.claim_next("pod2", {"cc"}))

    def test_stale_put_does_not_replace_owner(self):
        task = enqueue(self.first)
        old_body = self.s3.body
        old_etag = self.s3.etag.strip('"')
        self.first.claim_next("pod1", {"cc"})
        with self.assertRaises(ClientError):
            self.s3.put_object(Body=old_body, IfMatch=old_etag)
        self.assertEqual(self.first.get(task.task_id).pod_id, "pod1")

    def test_same_key_with_different_request_is_rejected(self):
        enqueue(self.first)
        with self.assertRaises(QueueConflictError):
            enqueue(self.first, output="s3://output/other")

    def test_overlapping_output_is_reserved_until_terminal(self):
        task = enqueue(self.first, output="s3://output/data")
        with self.assertRaises(QueueConflictError):
            enqueue(self.second, "other", "s3://output/data/child")
        self.first.claim_next("pod1", {"cc"})
        self.first.set_status(task.task_id, TaskStatus.FAILED, "Ошибка расчёта")
        retry = enqueue(self.second, "retry", "s3://output/data")
        self.assertNotEqual(task.task_id, retry.task_id)
        self.assertEqual(retry.status, TaskStatus.QUEUED)

    def test_queued_abort_cannot_be_claimed(self):
        task = enqueue(self.first)
        self.second.request_abort(task.task_id)
        self.assertIsNone(self.first.claim_next("pod1", {"cc"}))
        self.assertEqual(self.first.get(task.task_id).status, TaskStatus.ABORTED)

    def test_abort_and_progress_do_not_lose_each_other(self):
        task = enqueue(self.first)
        self.first.claim_next("pod1", {"cc"})
        self.concurrent(
            lambda: self.first.update_progress(task.task_id, 5, 2),
            lambda: self.second.request_abort(task.task_id),
        )
        state = self.first.get(task.task_id)
        self.assertEqual(state.processed_rows, 5)
        self.assertTrue(state.cancel_requested)
        self.assertFalse(self.first.prepare_completion(task.task_id, 5, 2))
        self.assertEqual(self.first.get(task.task_id).status, TaskStatus.ABORTED)

    def test_finalizing_rejects_abort(self):
        task = enqueue(self.first)
        self.first.claim_next("pod1", {"cc"})
        self.assertTrue(self.first.prepare_completion(task.task_id, 10, 2))
        with self.assertRaises(QueueConflictError):
            self.second.request_abort(task.task_id)
        self.first.set_status(task.task_id, TaskStatus.DONE)
        self.first.heartbeat(task.task_id)
        self.assertEqual(self.first.get(task.task_id).status, TaskStatus.DONE)

    def test_other_owner_cannot_change_progress(self):
        task = enqueue(self.first)
        self.first.claim_next("pod1", {"cc"})
        with self.assertRaises(PermissionError):
            self.second.update_progress(task.task_id, 10, 2)

    def test_expired_task_is_reserved_until_monitor_marks_failed(self):
        task = enqueue(self.first)
        self.first.claim_next("pod1", {"cc"})
        raw = json.loads(self.s3.body)
        raw["tasks"][task.task_id]["heartbeat_at"] = 1
        self.s3.body = json.dumps(raw).encode()
        self.assertFalse(self.first.executor_alive(self.first.get(task.task_id)))
        self.assertIsNone(self.second.claim_next("pod2", {"cc"}))
        with self.assertRaises(QueueConflictError):
            enqueue(self.second, "retry", task.s3_output_path)

    def test_retention_removes_only_old_terminal_tasks(self):
        task = enqueue(self.first)
        self.first.request_abort(task.task_id)
        pending = enqueue(self.first, "pending")
        raw = json.loads(self.s3.body)
        for record in raw["tasks"].values():
            record["created_at"] = record["updated_at"] = record["finished_at"] = 1
        self.s3.body = json.dumps(raw).encode()
        self.first.cleanup()
        self.assertIsNone(self.first.get(task.task_id))
        self.assertIsNotNone(self.first.get(pending.task_id))

    def test_idle_poll_and_noop_cleanup_do_not_write(self):
        writes = self.s3.writes
        self.first.claim_next("pod1", {"cc"})
        self.first.cleanup()
        self.assertEqual(self.s3.writes, writes)

    def test_queue_limit_and_model_filter(self):
        self.first = make_store(self.s3, max_pending_tasks=1)
        enqueue(self.first, model="unavailable")
        with self.assertRaises(QueueConflictError):
            enqueue(self.first, "other")
        self.assertIsNone(self.first.claim_next("pod1", {"cc"}))

    def test_initialization_preserves_existing_state(self):
        task = enqueue(self.first)
        before = self.s3.body
        self.second.initialize()
        self.assertEqual(self.s3.body, before)
        self.assertIsNotNone(self.second.get(task.task_id))

    def test_legacy_state_is_read_without_resetting_timestamps(self):
        task = enqueue(self.first)
        raw = json.loads(self.s3.body)
        del raw["revision"]
        del raw["tasks"][task.task_id]["created_at"]
        raw["tasks"][task.task_id]["started_at"] = 123
        self.s3.body = json.dumps(raw).encode()
        before = self.s3.body
        self.second.initialize()
        self.assertEqual(self.s3.body, before)
        self.assertEqual(self.second.get(task.task_id).created_at, 123)
        self.second.request_abort(task.task_id)
        self.assertEqual(self.second.get(task.task_id).created_at, 123)

    def test_initialization_lost_response_is_reconciled(self):
        s3 = FakeS3()
        s3.lose_next_response = True
        make_store(s3).initialize()
        self.assertEqual(s3.writes, 1)

    def test_simultaneous_initialization_uses_conditional_create(self):
        s3 = FakeS3()
        original_get = s3.get_object
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        calls = 0

        def get(**kwargs):
            nonlocal calls
            with lock:
                calls += 1
                first_read = calls <= 2
            if first_read:
                barrier.wait(timeout=5)
                raise s3.error(404)
            return original_get(**kwargs)

        s3.get_object = get
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(make_store(s3).initialize) for _ in range(2)]
            for future in futures:
                future.result(timeout=5)
        self.assertEqual(s3.writes, 1)
        self.assertTrue(all(match is None and create == "*" for match, create in s3.requests))

    def test_corrupt_file_is_never_recreated(self):
        self.s3.body = b"invalid json"
        before = self.s3.writes
        with self.assertRaises(ValueError):
            self.second.initialize()
        self.assertEqual(self.s3.writes, before)

    def test_access_denied_is_not_a_cas_conflict(self):
        self.s3.fail_next = 403
        with self.assertRaises(ClientError):
            enqueue(self.first)

    def test_server_error_is_retried(self):
        self.s3.fail_next = 503
        task = enqueue(self.first)
        self.assertEqual(self.first.get(task.task_id).status, TaskStatus.QUEUED)

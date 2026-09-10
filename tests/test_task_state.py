import copy
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from app.tasks.backends.base import TaskStorageBackend
from app.tasks.cancellation import CancellationRegistry
from app.tasks.manager import TaskAlreadyRunningError, TaskManager
from app.tasks.state import TaskState, TaskStatus, TaskStore


class MemoryBackend(TaskStorageBackend):
    """Хранилище с последовательными мутациями для тестов бизнес-логики, без имитации S3 lock."""

    def __init__(self):
        self.tasks = {}
        self.writes = 0
        self.mutex = threading.Lock()

    def read_state(self):
        with self.mutex:
            return copy.deepcopy(self.tasks)

    def mutate(self, fn):
        with self.mutex:
            self.tasks = copy.deepcopy(fn(copy.deepcopy(self.tasks)))
            self.writes += 1
            return copy.deepcopy(self.tasks)


class TaskStateTests(unittest.TestCase):
    def setUp(self):
        self.backend = MemoryBackend()
        self.owner = TaskStore(self.backend)
        self.remote = TaskStore(self.backend)
        self.clock = patch("app.tasks.state.time.time", return_value=1000.0)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.task = TaskState(
            task_id="task-1", model_name="cc", s3_input_path="s3://input/",
            s3_output_path="s3://output/", status=TaskStatus.RUNNING,
            total_rows=10, pod_id="pod-a", started_at=1000.0, updated_at=1000.0,
        )
        self.owner.add(self.task)

    def test_remote_abort_reaches_owner_without_reviving_stale_task(self):
        self.now.return_value = 1300.0
        result = self.remote.request_abort(self.task.task_id)
        self.assertEqual(result.status, TaskStatus.ABORTING)
        self.assertTrue(self.owner.cancellation_requested(self.task.task_id))
        self.assertEqual(result.heartbeat_at, 1000.0)
        self.assertEqual(result.updated_at, 1000.0)
        self.assertEqual(self.remote.get_all_active(), [])

    def test_active_reads_exclude_dead_executor_without_writing(self):
        writes = self.backend.writes
        self.assertEqual(len(self.remote.get_all_active()), 1)
        self.now.return_value = 1180.0
        self.assertEqual(self.remote.get_all_active(), [])
        self.assertIsNone(self.remote.get_active("pod-a"))
        self.assertEqual(self.remote.get(self.task.task_id).status, TaskStatus.RUNNING)
        self.assertEqual(self.backend.writes, writes)

    def test_heartbeat_restores_visibility_without_changing_progress(self):
        self.now.return_value = 1300.0
        self.owner.heartbeat(self.task.task_id)
        task = self.remote.get_all_active()[0]
        self.assertEqual(task.heartbeat_at, 1300.0)
        self.assertEqual(task.updated_at, 1000.0)
        self.assertEqual(task.processed_rows, 0)

    def test_legacy_record_uses_updated_at_for_liveness(self):
        self.backend.tasks[self.task.task_id].heartbeat_at = None
        self.backend.tasks[self.task.task_id].owner_id = None
        self.assertTrue(self.remote.executor_alive(self.remote.get(self.task.task_id)))
        self.now.return_value = 1300.0
        self.assertEqual(self.remote.get_all_active(), [])
        self.remote.request_abort(self.task.task_id)
        self.assertEqual(self.remote.get_all_active(), [])

    def test_foreign_process_cannot_refresh_progress_or_heartbeat(self):
        operations = [
            lambda: self.remote.heartbeat(self.task.task_id),
            lambda: self.remote.update_progress(self.task.task_id, 5, 1.0),
            lambda: self.remote.set_status(self.task.task_id, TaskStatus.FAILED),
            lambda: self.remote.prepare_completion(self.task.task_id, 10, 1.0),
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(PermissionError):
                    operation()

    def test_abort_accepted_before_finalization_prevents_success(self):
        self.remote.request_abort(self.task.task_id)
        self.assertFalse(self.owner.prepare_completion(self.task.task_id, 10, 2.0))
        task = self.owner.get(self.task.task_id)
        self.assertEqual(task.status, TaskStatus.ABORTED)
        self.assertEqual(task.processed_rows, 10)
        self.assertFalse(task.cancel_requested)

    def test_abort_after_finalization_does_not_turn_task_aborting(self):
        self.assertTrue(self.owner.prepare_completion(self.task.task_id, 10, 2.0))
        result = self.remote.request_abort(self.task.task_id)
        self.assertEqual(result.status, TaskStatus.FINALIZING)
        self.assertFalse(result.cancel_requested)
        self.owner.set_status(self.task.task_id, TaskStatus.DONE)
        self.assertEqual(self.owner.get(self.task.task_id).status, TaskStatus.DONE)

    def test_terminal_status_cannot_be_regressed_by_late_updates(self):
        self.owner.set_status(self.task.task_id, TaskStatus.ABORTED)
        before = self.owner.get(self.task.task_id)
        self.now.return_value = 1500.0
        self.remote.request_abort(self.task.task_id)
        self.owner.heartbeat(self.task.task_id)
        self.owner.update_progress(self.task.task_id, 5, 10.0)
        self.owner.set_status(self.task.task_id, TaskStatus.FAILED, "late failure")
        self.assertEqual(self.owner.get(self.task.task_id), before)

    def test_stale_record_does_not_block_new_process_on_same_pod(self):
        self.now.return_value = 1400.0
        replacement = copy.deepcopy(self.task)
        replacement.task_id = "task-2"
        self.assertIsNone(self.remote.try_add_if_no_active(replacement))
        self.assertEqual([task.task_id for task in self.remote.get_all_active()], ["task-2"])

    def test_live_record_blocks_second_start_on_same_pod(self):
        replacement = copy.deepcopy(self.task)
        replacement.task_id = "task-2"
        blocked = self.remote.try_add_if_no_active(replacement)
        self.assertEqual(blocked.task_id, self.task.task_id)
        self.assertIsNone(self.remote.get("task-2"))

    def test_busy_response_identifies_live_task_instead_of_stale_history(self):
        live = copy.deepcopy(self.task)
        live.task_id = "live-task"
        self.now.return_value = 1400.0
        self.assertIsNone(self.remote.try_add_if_no_active(live))
        candidate = copy.deepcopy(self.task)
        candidate.task_id = "candidate"
        blocked = self.owner.try_add_if_no_active(candidate)
        self.assertEqual(blocked.task_id, "live-task")

    def test_initial_heartbeat_is_stamped_after_waiting_for_lock(self):
        other = copy.deepcopy(self.task)
        other.task_id = "other-task"
        other.pod_id = "pod-b"
        mutate = self.backend.mutate

        def after_wait(fn):
            self.now.return_value = 1400.0
            return mutate(fn)

        with patch.object(self.backend, "mutate", side_effect=after_wait):
            self.owner.try_add_if_no_active(other)
        self.assertEqual(self.owner.get(other.task_id).heartbeat_at, 1400.0)
        self.assertTrue(self.owner.executor_alive(self.owner.get(other.task_id)))

    def test_concurrent_abort_and_completion_choose_one_boundary(self):
        barrier = threading.Barrier(2)

        def finish():
            barrier.wait(timeout=2)
            return self.owner.prepare_completion(self.task.task_id, 10, 1.0)

        def abort():
            barrier.wait(timeout=2)
            return self.remote.request_abort(self.task.task_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            completed = pool.submit(finish)
            cancelled = pool.submit(abort)
            prepared = completed.result(timeout=3)
            cancelled.result(timeout=3)
        task = self.owner.get(self.task.task_id)
        self.assertEqual(task.status, TaskStatus.FINALIZING if prepared else TaskStatus.ABORTED)

    def test_local_worker_blocks_new_infer_even_after_heartbeat_timeout(self):
        registry = CancellationRegistry()
        registry.create(self.task.task_id)
        manager = TaskManager(self.owner, registry, "pod-a")
        self.now.return_value = 1400.0
        with self.assertRaises(TaskAlreadyRunningError):
            manager.try_start_task("cc", "input", "output", 10)
        self.assertEqual(len(self.backend.tasks), 1)

    def test_manager_sends_persisted_cancellation_from_another_pod(self):
        registry = CancellationRegistry()
        manager = TaskManager(self.remote, registry, "pod-b")
        task = manager.request_abort(self.task.task_id)
        self.assertEqual(task.status, TaskStatus.ABORTING)
        self.assertEqual(registry.task_ids(), [])
        self.assertTrue(self.owner.cancellation_requested(self.task.task_id))


if __name__ == "__main__":
    unittest.main()

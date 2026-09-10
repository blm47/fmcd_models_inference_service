"""Интеграционные проверки task tracker с тестовым S3, хранящим данные в байтах."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from app.tasks.backends.s3_backend import S3LeaseLockBackend
from app.tasks.cancellation import CancellationRegistry
from app.tasks.manager import TaskManager
from app.tasks.state import TaskStatus, TaskStore
from test_s3_backend_lock import FakeClock, LOCK_KEY, MemoryS3, MONTH, STATE_KEY


class TaskS3IntegrationTests(unittest.TestCase):
    def test_three_pods_share_only_two_objects_across_task_lifecycle(self):
        fs = MemoryS3()
        clock = FakeClock()
        with ExitStack() as patches:
            patches.enter_context(patch("app.tasks.backends.s3_backend.time.time", side_effect=lambda: clock.wall))
            patches.enter_context(patch("app.tasks.backends.s3_backend.time.monotonic", side_effect=lambda: clock.monotonic))
            patches.enter_context(patch("app.tasks.backends.s3_backend.time.sleep", side_effect=clock.sleep))
            patches.enter_context(patch("app.tasks.backends.s3_backend.random.uniform", return_value=1))
            stores = []
            managers = []
            for pod in range(3):
                backend = S3LeaseLockBackend(fs, STATE_KEY, LOCK_KEY, retention_months=1)
                backend.initialize()
                store = TaskStore(backend)
                stores.append(store)
                managers.append(TaskManager(store, CancellationRegistry(), f"pod-{pod}"))

            tasks = [manager.try_start_task("cc", "input", f"output-{pod}", 100)
                     for pod, manager in enumerate(managers)]
            self.assertEqual(len(stores[0].get_all_active()), 3)
            managers[2].request_abort(tasks[0].task_id)
            self.assertTrue(stores[0].cancellation_requested(tasks[0].task_id))
            self.assertFalse(stores[0].prepare_completion(tasks[0].task_id, 20, 2.0))
            self.assertEqual(stores[1].get(tasks[0].task_id).status, TaskStatus.ABORTED)

            self.assertTrue(stores[1].prepare_completion(tasks[1].task_id, 100, 10.0))
            stores[1].set_status(tasks[1].task_id, TaskStatus.DONE)
            writes_before_reads = len(fs.writes)
            clock.advance(181)
            self.assertEqual(stores[0].get_all_active(), [])
            self.assertEqual(len(fs.writes), writes_before_reads)
            stores[2].heartbeat(tasks[2].task_id)
            self.assertEqual([task.task_id for task in stores[0].get_all_active()], [tasks[2].task_id])

            clock.advance(2 * MONTH)
            # Долго работающий worker сохраняется при обновлении heartbeat;
            # устаревшая история завершённых задач удаляется в той же мутации.
            stores[2].heartbeat(tasks[2].task_id)
            self.assertEqual(set(fs.get(STATE_KEY)["tasks"]), {tasks[2].task_id})
            stores[2].set_status(tasks[2].task_id, TaskStatus.FAILED, "worker failed")
            clock.advance(2 * MONTH)
            stores[0].cleanup()
            self.assertEqual(fs.get(STATE_KEY), {"tasks": {}})
            self.assertEqual(set(fs.objects), {STATE_KEY, LOCK_KEY})


if __name__ == "__main__":
    unittest.main()

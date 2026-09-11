"""Проверка одного worker на под и выполнения общей очереди несколькими подами."""

import threading
import types
import unittest
from unittest.mock import Mock, patch

from fake_s3 import FakeS3, enqueue, make_store
from logger_stub import make_logger

from app.tasks.state import TaskStatus
from app.tasks.worker import consume_queue


class ConsumerTests(unittest.TestCase):
    def test_two_pods_drain_queue_without_duplicate_execution(self):
        s3 = FakeS3()
        stores = [make_store(s3, poll_interval_sec=0.001) for _ in range(2)]
        stores[0].initialize()
        task_ids = {enqueue(stores[0], str(index)).task_id for index in range(6)}
        stop = threading.Event()
        all_done = threading.Event()
        running = {}
        completed = []
        lock = threading.Lock()
        first_workers = threading.Barrier(2)

        def worker(task, bundle, settings, store, shutdown, client, logger):
            with lock:
                self.assertNotIn(task.pod_id, running)
                running[task.pod_id] = task.task_id
                first_pair = len(completed) == 0
            if first_pair:
                first_workers.wait(timeout=5)
            store.prepare_completion(task.task_id, 10, 1)
            store.set_status(task.task_id, TaskStatus.DONE)
            with lock:
                del running[task.pod_id]
                completed.append(task.task_id)
                if len(completed) == 6:
                    all_done.set()

        logger = make_logger()
        threads = [
            threading.Thread(
                target=consume_queue,
                args=(
                    store,
                    {"cc": object()},
                    types.SimpleNamespace(task_store=store.config),
                    Mock(),
                    f"pod-{index}",
                    stop,
                    logger,
                ),
                daemon=True,
            )
            for index, store in enumerate(stores)
        ]
        with patch("app.tasks.worker.run_task", side_effect=worker):
            try:
                for thread in threads:
                    thread.start()
                self.assertTrue(all_done.wait(timeout=5))
            finally:
                stop.set()
                for thread in threads:
                    thread.join(timeout=5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(len(completed), 6)
        self.assertEqual(set(completed), task_ids)
        logger.error.assert_not_called()

    def test_shutdown_before_poll_does_not_claim(self):
        store = Mock()
        stop = threading.Event()
        stop.set()
        consume_queue(store, {}, Mock(), Mock(), "pod1", stop, Mock())
        store.claim_next.assert_not_called()

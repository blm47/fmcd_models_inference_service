import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

from logger_stub import make_logger

from app.tasks.state import TaskState, TaskStatus
from app.tasks.worker import _keep_heartbeat, run_task


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.task = TaskState(
            task_id="task-1",
            model_name="model",
            s3_input_path="s3://input/data",
            s3_output_path="s3://output/data",
            total_rows=4,
            status=TaskStatus.RUNNING,
        )
        self.settings = types.SimpleNamespace(
            inference=types.SimpleNamespace(chunk_size=2, infer_batch_size=1),
            task_store=types.SimpleNamespace(progress_update_interval_sec=120),
        )
        self.cancel_event = threading.Event()
        self.logger = make_logger()
        self.remote_cancel = False
        self.heartbeat_received = threading.Event()
        self.store = Mock()
        self.store.config.poll_interval_sec = 0.001
        self.store.heartbeat_interval_sec = 0.01
        self.store.heartbeat.side_effect = lambda task_id: self.heartbeat_received.set()
        self.store.cancellation_requested.side_effect = lambda task_id: self.remote_cancel
        self.store.update_progress.side_effect = self._update_progress
        self.store.set_status.side_effect = self._set_status
        self.store.prepare_completion.side_effect = self._prepare_completion
        self.writer = Mock()
        self.s3 = Mock()
        self.s3.prefix_has_results.return_value = False
        self.s3.get_writer.return_value = self.writer
        self.s3.iter_chunks.return_value = [[1, 2], [3, 4]]

        inference_module = types.ModuleType("app.models.inference")
        self.infer = Mock(side_effect=lambda chunk, bundle, batch_size, checkpoint: chunk)
        inference_module.run_inference_on_chunk = self.infer
        inference_patch = patch.dict(sys.modules, {"app.models.inference": inference_module})
        inference_patch.start()
        self.addCleanup(inference_patch.stop)

        # Сохраняем реальные потоки, чтобы проверить остановку heartbeat при выходе worker.
        thread_class = threading.Thread
        self.heartbeat_threads = []

        def make_thread(*args, **kwargs):
            thread = thread_class(*args, **kwargs)
            self.heartbeat_threads.append(thread)
            return thread

        thread_patch = patch("app.tasks.worker.threading.Thread", side_effect=make_thread)
        thread_patch.start()
        self.addCleanup(thread_patch.stop)

    def _update_progress(self, task_id, processed_rows, inference_elapsed_sec):
        self.task.processed_rows = processed_rows
        self.task.inference_elapsed_sec = inference_elapsed_sec

    def _set_status(self, task_id, status, error=None):
        self.task.status = status
        self.task.error = error

    def _prepare_completion(self, task_id, processed_rows, inference_elapsed_sec):
        self._update_progress(task_id, processed_rows, inference_elapsed_sec)
        self.task.status = TaskStatus.ABORTED if self.remote_cancel else TaskStatus.FINALIZING
        return not self.remote_cancel

    def run_worker(self):
        run_task(
            self.task,
            object(),
            self.settings,
            self.store,
            self.cancel_event,
            self.s3,
            self.logger,
        )

    def assert_worker_stopped(self):
        self.assertTrue(self.heartbeat_threads)
        self.assertTrue(all(not thread.is_alive() for thread in self.heartbeat_threads))

    def test_remote_abort_before_start_does_not_open_output(self):
        self.remote_cancel = True
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.ABORTED)
        self.assertEqual(self.task.processed_rows, 0)
        self.s3.get_writer.assert_not_called()
        self.infer.assert_not_called()
        self.assertFalse(self.cancel_event.is_set())
        self.assert_worker_stopped()

    def test_remote_abort_during_input_read_stops_before_inference(self):
        def chunks():
            self.remote_cancel = True
            yield [1, 2]

        self.s3.iter_chunks.return_value = chunks()
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.ABORTED)
        self.infer.assert_not_called()
        self.writer.write_chunk.assert_not_called()
        self.writer.close.assert_not_called()
        self.assert_worker_stopped()

    def test_remote_abort_during_inference_does_not_write_current_chunk(self):
        calls = 0

        def infer(chunk, bundle, batch_size, checkpoint):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.remote_cancel = True
            return chunk

        self.infer.side_effect = infer
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.ABORTED)
        self.assertEqual(self.task.processed_rows, 2)
        self.assertGreaterEqual(self.task.inference_elapsed_sec, 0)
        self.writer.write_chunk.assert_called_once_with([1, 2])
        self.writer.close.assert_not_called()
        self.store.prepare_completion.assert_not_called()
        self.assertFalse(self.cancel_event.is_set())
        self.assert_worker_stopped()

    def test_remote_abort_during_final_write_prevents_success_marker(self):
        def write(chunk):
            if chunk == [3, 4]:
                self.remote_cancel = True

        self.writer.write_chunk.side_effect = write
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.ABORTED)
        self.assertEqual(self.task.processed_rows, 4)
        self.store.prepare_completion.assert_called_once()
        self.writer.close.assert_not_called()
        self.assert_worker_stopped()

    def test_shutdown_during_final_write_sets_failed(self):
        self.s3.iter_chunks.return_value = [[1, 2]]
        self.writer.write_chunk.side_effect = lambda chunk: self.cancel_event.set()
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertIn("pod_shutdown", self.task.error)
        self.writer.close.assert_not_called()
        self.store.prepare_completion.assert_not_called()
        self.assert_worker_stopped()

    def test_shutdown_between_gpu_batches_stops_before_output(self):
        def infer(chunk, bundle, batch_size, checkpoint):
            self.cancel_event.set()
            checkpoint()
            self.fail("После shutdown новый GPU batch запускаться не должен")

        self.infer.side_effect = infer
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertIn("pod_shutdown", self.task.error)
        self.writer.write_chunk.assert_not_called()
        self.writer.close.assert_not_called()
        self.assert_worker_stopped()

    def test_shutdown_before_worker_start_sets_failed(self):
        self.cancel_event.set()
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.s3.get_writer.assert_not_called()
        self.assert_worker_stopped()

    def test_success_marker_is_written_only_after_completion_decision(self):
        def close():
            self.assertEqual(self.task.status, TaskStatus.FINALIZING)
            self.assertEqual(self.task.processed_rows, 4)

        self.writer.close.side_effect = close
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.DONE)
        self.assertEqual(self.writer.write_chunk.call_count, 2)
        self.writer.close.assert_called_once_with()
        self.assert_worker_stopped()

    def test_empty_input_finalizes_with_zero_rows_and_success_marker(self):
        self.s3.iter_chunks.return_value = []
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.DONE)
        self.assertEqual(self.task.processed_rows, 0)
        self.store.prepare_completion.assert_called_once_with(self.task.task_id, 0, 0.0)
        self.infer.assert_not_called()
        self.writer.write_chunk.assert_not_called()
        self.writer.close.assert_called_once_with()
        self.assert_worker_stopped()

    def test_done_write_is_retried_without_republishing_success_marker(self):
        calls = 0

        def set_status(task_id, status, error=None):
            nonlocal calls
            calls += 1
            self.assertEqual(status, TaskStatus.DONE)
            if calls == 1:
                raise OSError("temporary status PUT failure")
            self._set_status(task_id, status, error)

        self.store.set_status.side_effect = set_status
        with patch("app.tasks.worker.time.sleep"):
            self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.DONE)
        self.assertEqual(calls, 2)
        self.writer.close.assert_called_once_with()
        self.assert_worker_stopped()

    def test_published_result_is_never_marked_failed_on_status_delivery_error(self):
        self.store.set_status.side_effect = OSError("status S3 unavailable")
        self.writer.close.side_effect = lambda: self.cancel_event.set()
        with patch("app.tasks.worker.time.sleep"):
            self.run_worker()
        self.assertEqual(self.store.set_status.call_count, 1)
        for call in self.store.set_status.call_args_list:
            self.assertEqual(call.args[1], TaskStatus.DONE)
        self.writer.close.assert_called_once_with()
        self.assert_worker_stopped()

    def test_heartbeat_continues_while_inference_is_busy(self):
        def infer(chunk, bundle, batch_size, checkpoint):
            self.assertTrue(self.heartbeat_received.wait(timeout=2))
            self.assertEqual(self.task.processed_rows, 0)
            return chunk

        self.infer.side_effect = infer
        self.run_worker()

        self.store.heartbeat.assert_called_with(self.task.task_id)
        self.assertEqual(self.task.status, TaskStatus.DONE)
        self.assert_worker_stopped()

    def test_writer_initialization_failure_sets_failed_and_cleans_up(self):
        self.s3.get_writer.side_effect = RuntimeError("writer unavailable")
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertEqual(self.task.error, "writer unavailable")
        self.infer.assert_not_called()
        self.assert_worker_stopped()

    def test_missing_input_sets_failed_and_never_writes_success(self):
        self.s3.iter_chunks.side_effect = FileNotFoundError("input unavailable")
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertIn("s3_path_not_found", self.task.error)
        self.writer.close.assert_not_called()
        self.assert_worker_stopped()

    def test_failed_success_marker_sets_failed_instead_of_done(self):
        self.writer.close.side_effect = OSError("output unavailable")
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertEqual(self.task.error, "output unavailable")
        self.assert_worker_stopped()

    def test_failed_status_write_still_stops_heartbeat_and_releases_local_slot(self):
        def fail(*args):
            self.cancel_event.set()
            raise RuntimeError("inference failed")

        self.infer.side_effect = fail
        self.store.set_status.side_effect = OSError("status storage unavailable")

        self.run_worker()

        self.writer.close.assert_not_called()
        self.assert_worker_stopped()


class HeartbeatTests(unittest.TestCase):
    def test_terminal_task_signals_worker_to_stop(self):
        stop = Mock()
        stop.wait.return_value = False
        store = Mock()
        store.heartbeat.return_value = False
        lost = threading.Event()
        _keep_heartbeat("task-1", store, stop, Mock(), lost)
        self.assertTrue(lost.is_set())
        store.heartbeat.assert_called_once_with("task-1")

    def test_temporary_storage_failure_is_retried_until_stop(self):
        stop = Mock()
        stop.wait.side_effect = [False, False, True]
        store = Mock()
        store.heartbeat_interval_sec = 30
        store.heartbeat.side_effect = [OSError("temporary S3 failure"), None]
        logger = make_logger()

        _keep_heartbeat("task-1", store, stop, logger)

        self.assertEqual(store.heartbeat.call_count, 2)
        logger.error.assert_called_once()
        self.assertEqual(stop.wait.call_count, 3)
        stop.wait.assert_called_with(30)


if __name__ == "__main__":
    unittest.main()

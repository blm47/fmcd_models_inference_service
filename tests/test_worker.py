import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd
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
            task_store=types.SimpleNamespace(progress_update_interval_sec=120),
        )
        self.spec = types.SimpleNamespace(
            name="model",
            device="cpu",
            id_cols=("id",),
            infer_batch_size=1,
            parquet_read_chunk_size=2,
        )
        self.bundle = Mock()
        self.bundle.required_columns = ()
        self.bundle.output_columns = ("score",)
        self.bundle.spec = self.spec
        factory_patch = patch("app.tasks.worker.create_bundle", return_value=self.bundle)
        self.factory = factory_patch.start()
        self.addCleanup(factory_patch.stop)
        self.frames = [pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [3, 4]})]
        self.cancel_event = threading.Event()
        self.logger = make_logger()
        self.remote_cancel = False
        self.store = Mock()
        self.store.config.poll_interval_sec = 0.001
        self.store.config.heartbeat_interval_sec = 0.01
        self.store.cancellation_requested.side_effect = lambda task_id: self.remote_cancel
        self.store.update_progress.side_effect = self._update_progress
        self.store.set_status.side_effect = self._set_status
        self.store.prepare_completion.side_effect = self._prepare_completion
        self.writer = Mock()
        self.s3 = Mock()
        self.s3.prefix_has_results.return_value = False
        self.s3.get_writer.return_value = self.writer
        self.s3.iter_chunks.return_value = self.frames

        inference_module = types.ModuleType("app.models.inference")
        self.infer = Mock(side_effect=lambda chunk, bundle, batch_size, checkpoint: chunk)
        inference_module.run_inference_on_chunk = self.infer
        inference_patch = patch.dict(sys.modules, {"app.models.inference": inference_module})
        inference_patch.start()
        self.addCleanup(inference_patch.stop)

        validation_module = types.ModuleType("app.models.validation")
        self.validation = Mock(return_value=types.SimpleNamespace(is_valid=True, total_rows=4))
        validation_module.validate_input_parquet = self.validation
        validation_patch = patch.dict(sys.modules, {"app.models.validation": validation_module})
        validation_patch.start()
        self.addCleanup(validation_patch.stop)

        self.heartbeat_threads = []
        thread_class = threading.Thread

        def make_thread(*args, **kwargs):
            thread = thread_class(*args, **kwargs)
            self.heartbeat_threads.append(thread)
            return thread

        patcher = patch("app.tasks.worker.threading.Thread", side_effect=make_thread)
        patcher.start()
        self.addCleanup(patcher.stop)

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
            self.spec,
            self.settings,
            self.store,
            self.cancel_event,
            self.s3,
            self.logger,
        )

    def assert_worker_stopped(self):
        self.assertTrue(all(not thread.is_alive() for thread in self.heartbeat_threads))
        if self.factory.called:
            self.bundle.close.assert_called_once_with()

    def test_utilization_is_not_started_by_default(self):
        with patch("app.tasks.worker.UtilizationSampler") as sampler:
            self.run_worker()
        sampler.assert_not_called()
        self.assert_worker_stopped()

    def test_utilization_stops_after_model_close_on_success_failure_and_cancel(self):
        for outcome in ("success", "failure", "cancel", "close_failure"):
            with self.subTest(outcome=outcome):
                self.bundle.reset_mock()
                self.task.calc_utilization = True
                self.remote_cancel = False
                self.cancel_event.clear()

                def load(outcome=outcome):
                    if outcome == "failure":
                        raise RuntimeError("load failed")
                    if outcome == "cancel":
                        self.remote_cancel = True

                self.bundle.load.side_effect = load
                self.bundle.close.side_effect = (
                    RuntimeError("close failed") if outcome == "close_failure" else None
                )
                with patch("app.tasks.worker.UtilizationSampler") as sampler:
                    collector = sampler.return_value
                    collector.snapshot.return_value = {"metrics_util_cpu_mean": 42}
                    collector.stop.side_effect = lambda: self.bundle.close.assert_called_once()
                    self.run_worker()
                    sampler.assert_called_once_with(
                        self.task.task_id, self.spec.device, self.logger
                    )
                    collector.start.assert_called_once()
                    collector.stop.assert_called_once()
                    self.assertTrue(
                        any(
                            '"metrics_util_cpu_mean": 42' in call.args[0]
                            for call in self.logger.info.call_args_list
                        )
                    )
                self.assert_worker_stopped()

    def test_partial_load_failure_closes_without_validating_input(self):
        self.bundle.load.side_effect = RuntimeError("partial load")
        self.run_worker()
        self.assertEqual(self.task.error, "partial load")
        self.validation.assert_not_called()
        self.assert_worker_stopped()

    def test_close_error_preserves_done_and_stops_pod(self):
        self.bundle.close.side_effect = RuntimeError("close failed")
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.DONE)
        self.assertTrue(self.cancel_event.is_set())
        self.assert_worker_stopped()

    def test_close_error_does_not_mask_load_error(self):
        self.bundle.load.side_effect = RuntimeError("load failed")
        self.bundle.close.side_effect = RuntimeError("close failed")
        self.run_worker()
        self.assertEqual(self.task.error, "load failed")
        self.assertTrue(self.cancel_event.is_set())
        self.assert_worker_stopped()

    def test_cancel_during_load_before_validation(self):
        def load():
            self.remote_cancel = True

        self.bundle.load.side_effect = load
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.ABORTED)
        self.validation.assert_not_called()
        self.assert_worker_stopped()

    def test_load_precedes_validation_and_uses_model_sizes(self):
        def validate(*args):
            self.bundle.load.assert_called_once_with()
            return types.SimpleNamespace(is_valid=True, total_rows=4)

        self.validation.side_effect = validate
        self.run_worker()
        self.s3.iter_chunks.assert_called_once_with(self.task.s3_input_path, 2)
        self.assertEqual(self.infer.call_args.args[2], 1)
        self.assert_worker_stopped()

    def test_row_count_change_prevents_success(self):
        self.validation.return_value.total_rows = 5
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.writer.close.assert_not_called()
        self.assert_worker_stopped()

    def test_dtype_change_between_chunks_prevents_second_write(self):
        def infer(chunk, *args):
            return chunk.astype(str) if chunk.id.iloc[0] == 3 else chunk

        self.infer.side_effect = infer
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.writer.write_chunk.assert_called_once()
        self.writer.close.assert_not_called()
        self.assert_worker_stopped()

    def test_chunk_write_failure_closes_model(self):
        self.writer.write_chunk.side_effect = OSError("chunk write failed")
        self.run_worker()
        self.assertEqual(self.task.error, "chunk write failed")
        self.assert_worker_stopped()

    def test_remote_abort_before_start_does_not_open_output(self):
        self.remote_cancel = True
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.ABORTED)
        self.assertEqual(self.task.processed_rows, 0)
        self.s3.get_writer.assert_not_called()
        self.infer.assert_not_called()
        self.assertFalse(self.cancel_event.is_set())
        self.assert_worker_stopped()

    def test_missing_columns_fail_before_inference_or_output(self):
        self.validation.return_value = types.SimpleNamespace(
            is_valid=False, missing_columns=["income"], total_rows=0
        )
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertIn("missing_columns", self.task.error)
        self.assertIn("income", self.task.error)
        self.infer.assert_not_called()
        self.s3.get_writer.assert_not_called()
        self.assert_worker_stopped()

    def test_missing_input_fails_in_worker(self):
        self.validation.side_effect = FileNotFoundError("input")
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.FAILED)
        self.assertIn("s3_path_not_found", self.task.error)
        self.s3.get_writer.assert_not_called()
        self.assert_worker_stopped()

    def test_worker_saves_row_count_after_validation(self):
        self.task.total_rows = None
        self.run_worker()
        self.store.set_total_rows.assert_called_once_with("task-1", 4)
        self.assertEqual(self.task.total_rows, 4)
        self.assertEqual(self.task.status, TaskStatus.DONE)

    def test_cancel_during_validation_does_not_open_output(self):
        def validate(*args):
            self.remote_cancel = True
            return types.SimpleNamespace(is_valid=True, total_rows=4)

        self.validation.side_effect = validate
        self.run_worker()
        self.assertEqual(self.task.status, TaskStatus.ABORTED)
        self.store.set_total_rows.assert_not_called()
        self.s3.get_writer.assert_not_called()
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
        pd.testing.assert_frame_equal(self.writer.write_chunk.call_args.args[0], self.frames[0])
        self.writer.write_chunk.assert_called_once()
        self.writer.close.assert_not_called()
        self.store.prepare_completion.assert_not_called()
        self.assertFalse(self.cancel_event.is_set())
        self.assert_worker_stopped()

    def test_remote_abort_during_final_write_prevents_success_marker(self):
        def write(chunk):
            if chunk.equals(self.frames[1]):
                self.remote_cancel = True

        self.writer.write_chunk.side_effect = write
        self.run_worker()

        self.assertEqual(self.task.status, TaskStatus.ABORTED)
        self.assertEqual(self.task.processed_rows, 4)
        self.store.prepare_completion.assert_not_called()
        self.writer.close.assert_not_called()
        self.assert_worker_stopped()

    def test_shutdown_during_final_write_sets_failed(self):
        self.s3.iter_chunks.return_value = self.frames[:1]
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
        self.validation.return_value.total_rows = 0
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

    def test_heartbeat_runs_during_load_and_stops_after_close(self):
        received = threading.Event()
        self.store.heartbeat.side_effect = lambda task_id: received.set() or True

        def load():
            self.assertTrue(received.wait(timeout=2))

        self.bundle.load.side_effect = load
        real_thread = threading.Thread
        threads = []

        def make_thread(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            threads.append(thread)
            return thread

        with patch("app.tasks.worker.threading.Thread", side_effect=make_thread):
            self.run_worker()
        self.assertEqual(len(threads), 1)
        self.assertFalse(threads[0].is_alive())
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

    def test_failed_status_write_still_releases_local_slot(self):
        def fail(*args):
            self.cancel_event.set()
            raise RuntimeError("inference failed")

        self.infer.side_effect = fail
        self.store.set_status.side_effect = OSError("status storage unavailable")

        self.run_worker()

        self.writer.close.assert_not_called()
        self.assert_worker_stopped()


class HeartbeatTests(unittest.TestCase):
    def test_terminal_task_signals_loss_and_stops_loop(self):
        store, stop, logger = Mock(), Mock(), make_logger()
        store.config.heartbeat_interval_sec = 30
        stop.wait.return_value = False
        store.heartbeat.return_value = False
        lost = threading.Event()
        _keep_heartbeat("task", store, stop, lost, logger)
        self.assertTrue(lost.is_set())
        store.heartbeat.assert_called_once_with("task")

    def test_storage_error_is_logged_and_retried(self):
        store, stop, logger = Mock(), Mock(), make_logger()
        store.config.heartbeat_interval_sec = 30
        stop.wait.side_effect = [False, False, True]
        store.heartbeat.side_effect = [OSError("temporary failure"), True]
        lost = threading.Event()
        _keep_heartbeat("task", store, stop, lost, logger)
        self.assertFalse(lost.is_set())
        self.assertEqual(store.heartbeat.call_count, 2)
        logger.error.assert_called_once()
        stop.wait.assert_called_with(30)


if __name__ == "__main__":
    unittest.main()

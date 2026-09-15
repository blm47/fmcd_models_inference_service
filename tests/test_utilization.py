"""
Посекундные замеры, точная статистика и сохранение утилизации через CAS.
"""

import statistics
import sys
import types
import unittest
from unittest.mock import Mock, patch

from fake_s3 import FakeS3, make_store
from logger_stub import make_logger

from app.tasks.state import TaskStatus
from app.tasks.utilization import ResourceProbe, UtilizationSampler, UtilizationStatistics


class UtilizationTests(unittest.TestCase):
    def test_exact_statistics_include_zero_and_skip_missing_samples(self):
        stats = UtilizationStatistics()
        values = [10, 0, 3, 9, 3, 100]
        for i, value in enumerate(values):
            stats.add({"cpu": value, "ram_mb": None, "gpu": float("nan")})
            snapshot = stats.snapshot()
            self.assertEqual(
                snapshot["metrics_util_cpu_median"], statistics.median(values[: i + 1])
            )
        self.assertEqual(snapshot["metrics_util_cpu_min"], 0)
        self.assertEqual(snapshot["metrics_util_cpu_max"], 100)
        self.assertEqual(snapshot["metrics_util_cpu_mean"], sum(values) / len(values))
        self.assertEqual(snapshot["metrics_util_cpu_samples"], 6)
        self.assertEqual(snapshot["metrics_util_ram_mb_samples"], 0)
        self.assertIsNone(snapshot["metrics_util_gpu_mean"])

    def test_sampler_uses_one_second_schedule_and_closes_probe(self):
        sampler = UtilizationSampler("task", "cpu", make_logger())
        sampler.probe = Mock()
        sampler.probe.sample.return_value = {"cpu": 12}
        sampler.stop_event = Mock()
        sampler.stop_event.wait.side_effect = [False, False, True]
        with patch("app.tasks.utilization.time.monotonic", side_effect=[0, 0, 1, 1, 2, 2]):
            sampler._run()
        self.assertEqual(
            [call.args[0] for call in sampler.stop_event.wait.call_args_list], [1, 1, 1]
        )
        self.assertEqual(sampler.snapshot()["metrics_util_cpu_samples"], 2)
        sampler.probe.close.assert_called_once()

    def test_probe_uses_process_and_device_uuid_without_initializing_cuda(self):
        nvml = Mock()
        nvml.nvmlDeviceGetUtilizationRates.return_value.gpu = 75
        nvml.nvmlDeviceGetMemoryInfo.return_value.used = 4096
        nvml.nvmlDeviceGetMemoryInfo.return_value.total = 8192
        torch = Mock()
        torch.cuda.is_initialized.return_value = False
        torch.cuda.get_device_properties.return_value.uuid = "GPU-selected"
        process = Mock()
        process.cpu_percent.return_value = 123
        process.memory_info.return_value.rss = 1000
        with patch.dict(
            sys.modules,
            {
                "psutil": types.SimpleNamespace(
                    Process=lambda: process,
                    virtual_memory=lambda: types.SimpleNamespace(total=2000),
                ),
                "pynvml": nvml,
                "torch": torch,
            },
        ):
            probe = ResourceProbe("cuda:2", make_logger())
            probe.start()
            self.assertEqual(probe.sample(), {"cpu": 123, "ram_mb": 0.001, "ram_pct": 50.0})
            torch.cuda.get_device_properties.assert_not_called()
            torch.cuda.is_initialized.return_value = True
            self.assertEqual(probe.sample()["gpu_ram_mb"], 0.004096)
            self.assertEqual(probe.sample()["gpu_ram_pct"], 50.0)
            nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-selected")
            torch.cuda.get_device_properties.assert_called_once_with("cuda:2")
            probe.close()
        nvml.nvmlShutdown.assert_called_once()

    def test_missing_gpu_sensor_keeps_cpu_and_ram(self):
        probe = ResourceProbe("cuda", make_logger())
        probe.process = Mock()
        probe.process.cpu_percent.return_value = 0
        probe.process.memory_info.return_value.rss = 2048
        self.assertEqual(probe.sample(), {"cpu": 0, "ram_mb": 0.002048})


class UtilizationStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = make_store(FakeS3())
        self.store.initialize()
        self.task = self.store.enqueue("key", "cc", "s3://in/data", "s3://out/data", None)
        self.store.claim_next("pod", {"cc"})
        self.collector = Mock()
        self.stats = UtilizationStatistics()
        self.collector.snapshot.side_effect = self.stats.snapshot
        patcher = patch("app.tasks.state.UtilizationSampler", return_value=self.collector)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store.start_utilization(self.task.task_id, "cpu")

    def test_each_task_write_includes_latest_snapshot_without_resetting_timeouts(self):
        before = self.store.get(self.task.task_id)
        self.stats.add({"cpu": 10, "ram_mb": 100})
        self.store.heartbeat(self.task.task_id)
        first = self.store.get(self.task.task_id)
        self.assertEqual(first.metrics_util_cpu_mean, 10)
        self.assertEqual(first.updated_at, before.updated_at)
        self.stats.add({"cpu": 30, "ram_mb": 300})
        self.store.set_total_rows(self.task.task_id, 4)
        self.assertEqual(self.store.get(self.task.task_id).metrics_util_ram_mb_median, 200)
        self.store.update_progress(self.task.task_id, 4, 1)
        self.store.prepare_completion(self.task.task_id, 4, 1)
        self.store.set_status(self.task.task_id, TaskStatus.DONE)
        terminal = self.store.get(self.task.task_id)
        self.stats.add({"cpu": 100})
        self.store.finish_utilization(self.task.task_id)
        final = self.store.get(self.task.task_id)
        self.assertEqual(final.status, TaskStatus.DONE)
        self.assertEqual(final.finished_at, terminal.finished_at)
        self.assertEqual(final.updated_at, terminal.updated_at)
        self.assertEqual(final.metrics_util_cpu_max, 100)
        self.assertEqual(final.metrics_util_cpu_samples, 3)
        self.assertEqual(self.store._utilization, {})
        self.collector.stop.assert_called_once()

    def test_metrics_do_not_revive_task_failed_by_another_pod(self):
        monitor = make_store(self.store.client)
        self.store.set_status(self.task.task_id, TaskStatus.FAILED, error="original")
        self.stats.add({"cpu": 42})
        self.store.finish_utilization(self.task.task_id)
        task = monitor.get(self.task.task_id)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.error, "original")
        self.assertEqual(task.metrics_util_cpu_mean, 42)

    def test_statistics_are_not_written_by_sampling_alone(self):
        self.stats.add({"cpu": 12})
        self.assertIsNone(self.store.get(self.task.task_id).metrics_util_cpu_mean)

    def test_cas_retry_does_not_duplicate_samples(self):
        self.stats.add({"cpu": 10})
        self.store.client.fail_next = 412
        self.store.heartbeat(self.task.task_id)
        task = self.store.get(self.task.task_id)
        self.assertEqual(task.metrics_util_cpu_samples, 1)
        self.assertEqual(task.metrics_util_cpu_mean, 10)

    def test_sampler_is_detached_even_if_final_write_fails(self):
        with patch.object(self.store, "_mutate", side_effect=RuntimeError("S3 failed")):
            with self.assertRaisesRegex(RuntimeError, "S3 failed"):
                self.store.finish_utilization(self.task.task_id)
        self.assertEqual(self.store._utilization, {})
        self.collector.stop.assert_called_once()

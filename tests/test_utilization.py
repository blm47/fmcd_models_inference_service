"""
Посекундные замеры, точная статистика и сохранение утилизации через CAS.
"""

import json
import statistics
import sys
import types
import unittest
from unittest.mock import Mock, patch

from fake_s3 import FakeS3, make_store
from logger_stub import make_logger

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
    def test_flag_survives_claim_and_metrics_are_not_serialized(self):
        store = make_store(FakeS3())
        store.initialize()
        task = store.enqueue(
            "key", "cc", "s3://in/data", "s3://out/data", None, calc_utilization=True
        )
        worker_store = make_store(store.backend.client)
        claimed = worker_store.claim_next("pod", {"cc"})
        self.assertTrue(claimed.calc_utilization)
        worker_store.update_progress(task.task_id, 1, 0.5)
        fields = json.loads(store.backend.client.body)["tasks"][task.task_id]
        self.assertFalse(any(key.startswith("metrics_util_") for key in fields))

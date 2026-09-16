"""
Посекундные измерения ресурсов и точная статистика одной задачи.
"""

import heapq
import math
import sys
import threading
import time
from pathlib import Path

RESOURCES = ("cpu", "ram_pct", "ram_mb", "gpu", "gpu_ram_pct", "gpu_ram_mb")
STATISTICS = ("min", "max", "mean", "median")


class RunningStatistics:
    """
    Две кучи дают точную медиану без пересортировки истории при каждой записи.
    """

    def __init__(self):
        self.lower = []
        self.upper = []
        self.total = 0.0
        self.minimum = None
        self.maximum = None

    def add(self, value):
        if value is None or not math.isfinite(value) or value < 0:
            return
        value = float(value)
        heapq.heappush(self.lower, -value)
        heapq.heappush(self.upper, -heapq.heappop(self.lower))
        if len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def snapshot(self):
        count = len(self.lower) + len(self.upper)
        if not count:
            return dict.fromkeys(STATISTICS) | {"samples": 0}
        median = -self.lower[0]
        if len(self.lower) == len(self.upper):
            median = (median + self.upper[0]) / 2
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / count,
            "median": median,
            "samples": count,
        }


class UtilizationStatistics:
    def __init__(self):
        self._lock = threading.Lock()
        self._resources = {resource: RunningStatistics() for resource in RESOURCES}

    def add(self, values):
        with self._lock:
            for resource, stats in self._resources.items():
                stats.add(values.get(resource))

    def snapshot(self):
        with self._lock:
            return {
                f"metrics_util_{resource}_{stat}": value
                for resource, stats in self._resources.items()
                for stat, value in stats.snapshot().items()
            }


class ResourceProbe:
    """
    CPU/RAM процесса; GPU/GPU_RAM выбранного устройства по NVML.
    """

    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.process = None
        self.nvml = None
        self.handle = None
        self.ram_limit = None
        self._warned = set()

    def _warn(self, resource, exc):
        if resource not in self._warned:
            self.logger.warn(f"Утилизация {resource} недоступна: {exc}")
            self._warned.add(resource)

    def start(self):
        try:
            import psutil

            self.process = psutil.Process()
            self.process.cpu_percent(None)
            limits = [psutil.virtual_memory().total]
            for path in (
                "/sys/fs/cgroup/memory.max",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            ):
                try:
                    value = Path(path).read_text(encoding="utf-8").strip()
                    if value.isdigit() and int(value) > 0:
                        limits.append(int(value))
                except OSError:
                    pass
            self.ram_limit = min(limits)
        except Exception as exc:
            self._warn("CPU/RAM", exc)
        if self.device.split(":", 1)[0] == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml = pynvml
            except Exception as exc:
                self._warn("GPU", exc)

    def sample(self):
        values = {}
        if self.process is not None:
            try:
                values["cpu"] = self.process.cpu_percent(None)
            except Exception as exc:
                self._warn("CPU", exc)
            try:
                rss = self.process.memory_info().rss
                values["ram_mb"] = rss / 1_000_000
                if self.ram_limit:
                    values["ram_pct"] = 100 * rss / self.ram_limit
            except Exception as exc:
                self._warn("RAM", exc)
        if self.nvml is not None:
            if self.handle is None:
                # UUID PyTorch учитывает CUDA_VISIBLE_DEVICES; CUDA ради метрик не запускаем.
                torch = sys.modules.get("torch")
                try:
                    cuda = getattr(torch, "cuda", None)
                    if cuda is None or not getattr(cuda, "is_initialized", lambda: False)():
                        return values
                    uuid = cuda.get_device_properties(self.device).uuid
                    self.handle = self.nvml.nvmlDeviceGetHandleByUUID(
                        uuid.decode() if isinstance(uuid, bytes) else str(uuid)
                    )
                except Exception as exc:
                    self._warn("GPU UUID", exc)
                    return values
            try:
                values["gpu"] = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            except Exception as exc:
                self._warn("GPU", exc)
            try:
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                values["gpu_ram_mb"] = memory.used / 1_000_000
                if memory.total > 0:
                    values["gpu_ram_pct"] = 100 * memory.used / memory.total
            except Exception as exc:
                self._warn("GPU RAM", exc)
        return values

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception as exc:
                self._warn("NVML shutdown", exc)
            finally:
                self.nvml = None


class UtilizationSampler:
    """
    Снимает показатели раз в секунду, без записи в S3 из потока измерений.
    """

    def __init__(self, task_id, device, logger):
        self.statistics = UtilizationStatistics()
        self.probe = ResourceProbe(device, logger)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"utilization-{task_id}", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            self.probe.start()
            deadline = time.monotonic() + 1.0
            while not self.stop_event.wait(max(0, deadline - time.monotonic())):
                self.statistics.add(self.probe.sample())
                deadline += 1.0
                if deadline <= time.monotonic():
                    deadline = time.monotonic() + 1.0
        finally:
            self.probe.close()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def snapshot(self):
        return self.statistics.snapshot()

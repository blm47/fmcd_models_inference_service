import time
import logging

logger = logging.getLogger(__name__)


class StageTimer:
    """
    Простой аккумулятор времени по стадиям пайплайна инференса.
    Вставляется в worker.py вокруг каждой стадии обработки чанка,
    чтобы понять, где реально уходит время: чтение S3, препроцессинг
    (pandas -> FMCD), GPU-инференс, запись в S3.
    """
    def __init__(self):
        self.totals: dict[str, float] = {}

    def track(self, stage: str):
        return _StageContext(self, stage)

    def report(self, total_rows: int):
        grand_total = sum(self.totals.values())
        logger.warning("=== Разбивка времени по стадиям (total_rows=%d) ===", total_rows)
        for stage, seconds in sorted(self.totals.items(), key=lambda x: -x[1]):
            pct = 100 * seconds / grand_total if grand_total else 0
            logger.warning("  %-20s %8.1f сек  (%5.1f%%)", stage, seconds, pct)
        logger.warning("  %-20s %8.1f сек", "ИТОГО", grand_total)


class _StageContext:
    def __init__(self, timer: StageTimer, stage: str):
        self.timer = timer
        self.stage = stage

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self._start
        self.timer.totals[self.stage] = self.timer.totals.get(self.stage, 0.0) + elapsed


# ПРИМЕР ИНТЕГРАЦИИ в worker.py:
#
# stage_timer = StageTimer()
#
# for chunk_idx, ... in enumerate(...):
#     with stage_timer.track("s3_read"):
#         df_chunk = next(chunk_iterator)
#
#     with stage_timer.track("preprocess_fmcd"):
#         fmcd_batch = pandas_chunk_to_fmcd_batch(df_chunk, ...)
#
#     with stage_timer.track("gpu_inference"):
#         predictions = run_inference(model, fmcd_batch, gpu_batch_size)
#
#     with stage_timer.track("s3_write"):
#         writer.write_chunk(result_df)
#
# stage_timer.report(total_rows=1_000_000)

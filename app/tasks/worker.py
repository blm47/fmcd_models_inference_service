"""
Основной фоновый цикл обработки задачи инференса.

ВАЖНО: s3_input_path/s3_output_path в TaskState - это ПРЕФИКСЫ, а не
одиночные файлы. Входной префикс - папка со Spark-выгрузкой
(_SUCCESS + part-*.parquet), выходной префикс - папка, куда пишутся
part-файлы по chunk_size строк каждый + свой _SUCCESS в конце
(см. app/storage/s3_client.py: ParquetPrefixWriter).

Поток выполнения:
  1. Читаем все part-файлы под входным префиксом как единый dataset,
     чанками по chunk_size строк (из конфига, по умолчанию 100_000) -
     границы чанков НЕ привязаны к границам исходных part-файлов Spark.
  2. На каждый чанк: run_inference_on_chunk (models/inference.py).
  3. Пишем результат чанка как отдельный part-файл под выходным префиксом
     сразу же (инкрементально, не копим весь результат в памяти).
  4. Между чанками: проверяем cancellation-флаг -> если взведён, прерываем
     и помечаем задачу ABORTED (уже записанные part-файлы остаются как
     есть, _SUCCESS в выходной префикс в этом случае НЕ кладётся - это
     сигнал для downstream, что выгрузка неполная).
  5. Прогресс (processed_rows, inference_elapsed_sec) обновляем в TaskStore
     не чаще, чем раз в progress_update_interval_sec секунд, чтобы не
     создавать лишний lock-contention при мелких чанках.

Важно: inference_elapsed_sec копит время ТОЛЬКО forward-pass
(models/inference.py), не включая время чтения чанка из S3 или записи
part-файла в S3 - по требованию "ETA только по GPU-инференсу".
"""

import logging
import time

from app.core.config import Settings
from app.models.inference import run_inference_on_chunk
from app.models.registry import ModelBundle
from app.storage.s3_client import S3Client
from app.tasks.cancellation import CancellationRegistry
from app.tasks.state import TaskState, TaskStatus, TaskStore

logger = logging.getLogger(__name__)


def run_task(
    task: TaskState,
    bundle: ModelBundle,
    settings: Settings,
    store: TaskStore,
    cancellation: CancellationRegistry,
    s3_client: S3Client,
) -> None:
    task_id = task.task_id
    processed_rows = 0
    inference_elapsed_sec = 0.0
    last_progress_flush = time.time()
    writer = s3_client.get_writer(task.s3_output_path)
    success_written = False

    try:
        for df_chunk in s3_client.iter_chunks(task.s3_input_path, settings.inference.chunk_size):
            if cancellation.is_cancelled(task_id):
                logger.info("Задача %s: получен сигнал отмены, останавливаемся", task_id)
                store.set_status(task_id, TaskStatus.ABORTED)
                return

            t0 = time.perf_counter()
            result_chunk = run_inference_on_chunk(
                df_chunk, bundle, settings.inference.infer_batch_size
            )
            inference_elapsed_sec += time.perf_counter() - t0

            writer.write_chunk(result_chunk)
            processed_rows += len(df_chunk)

            if time.time() - last_progress_flush >= settings.task.progress_update_interval_sec:
                store.update_progress(task_id, processed_rows, inference_elapsed_sec)
                last_progress_flush = time.time()

        writer.close()
        success_written = True

        store.update_progress(task_id, processed_rows, inference_elapsed_sec)
        store.set_status(task_id, TaskStatus.DONE)
        logger.info("Задача %s завершена: processed_rows=%d", task_id, processed_rows)

    except Exception as exc:
        logger.exception("Задача %s упала с ошибкой", task_id)
        store.set_status(task_id, TaskStatus.FAILED, error=str(exc))
    finally:
        if not success_written:
            logger.warning(
                "Задача %s: _SUCCESS в выходной префикс не записан (частичная/прерванная выгрузка)",
                task_id,
            )
        cancellation.cleanup(task_id)

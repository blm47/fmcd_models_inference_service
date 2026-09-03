"""
Основной фоновый цикл обработки задачи инференса.

Поток выполнения:
  1. Читаем input parquet чанками по chunk_size (из конфига, по умолчанию 100_000).
  2. На каждый чанк: run_inference_on_chunk (models/inference.py).
  3. Пишем результат чанка сразу в output parquet (инкрементально, не копим в памяти).
  4. Между чанками: проверяем cancellation-флаг -> если взведён, прерываем и
     помечаем задачу ABORTED (частично записанный output остаётся как есть).
  5. Прогресс (processed_rows, inference_elapsed_sec) обновляем в TaskStore не
     чаще, чем в progress_update_interval_sec секунд (не после каждого чанка),
     чтобы не создавать лишний lock-contention при мелких чанках.

Важно: inference_elapsed_sec копит время ТОЛЬКО forward-pass (models/inference.py),
не включая время чтения чанка из S3 или записи в S3.
"""

import logging
import time

from app.core.config import Settings
from app.core.resource_utilization_monitoring import ResourceMonitoring
from app.models.inference import run_inference_on_chunk
from app.models.registry import ModelBundle
from app.storage.s3_client import S3Client
from app.tasks.cancellation import CancellationRegistry
from app.tasks.state import TaskState, TaskStatus, TaskStore

# logger = logging.getLogger(__name__)


def run_task(
    task: TaskState,
    bundle: ModelBundle,
    settings: Settings,
    store: TaskStore,
    cancellation: CancellationRegistry,
    s3_client: S3Client,
    logger: logging.Logger
) -> None:
    task_id = task.task_id
    processed_rows = 0
    inference_elapsed_sec = 0.0
    last_progress_flush = time.time()
    writer = s3_client.get_writer(task.s3_output_path)
    success_writen = False

    # на проде выключить!
    utilization_monitor = ResourceMonitoring(f'{task_id}.csv', interval=1)

    try:
        logger.info(f"Задача {task_id} начата: total_rows={task.total_rows:,.0f}")
        utilization_monitor.start()

        for df_chunk in s3_client.iter_chunks(task.s3_input_path, settings.inference.chunk_size):
            if cancellation.is_cancelled(task_id):
                logger.info(f"Задача {task_id}: получен сигнал отмены, останавливаемся")
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
        success_writen = True

        store.update_progress(task_id, processed_rows, inference_elapsed_sec)
        store.set_status(task_id, TaskStatus.DONE)
        logger.info(f"Задача {task_id} завершена: processed_rows={processed_rows:,.0f}")
        utilization_monitor.stop()

    except FileNotFoundError as exc:
        logger.exception(f"Задача {task_id}: входной префикс стал недоступен в S3")
        store.set_status(task_id, TaskStatus.FAILED, error=f"s3_path_not_found: {exc}")
    except Exception as exc:
        logger.exception(f"Задача {task_id} упала с ошибкой")
        store.set_status(task_id, TaskStatus.FAILED, error=str(exc))
    finally:
        if not success_writen:
            logger.warn(f"Задача {task_id}: _SUCCESS в выходной префикс не записан (частичная или прерванная загрузка)")
        cancellation.cleanup(task_id)

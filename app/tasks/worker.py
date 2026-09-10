"""Фоновый инференс с отменой через S3 и отдельным heartbeat.

Отмена кооперативная: уже начатый GPU forward-pass завершается, после чего
его результат не записывается, если поступила отмена. Перед записью _SUCCESS
TaskStore фиксирует переход в FINALIZING или принимает последнюю отмену.
Heartbeat показывает доступность исполнителя, а не движение прогресса.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from app.tasks.cancellation import CancellationRegistry
from app.tasks.state import TaskState, TaskStatus, TaskStore

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.models.registry import ModelBundle
    from app.storage.s3_client import S3Client


def _keep_heartbeat(
    task_id: str,
    store: TaskStore,
    stop: threading.Event,
    logger: logging.Logger,
) -> None:
    """Не блокирует heartbeat на долгом чтении, forward-pass или записи чанка."""
    while not stop.wait(store.heartbeat_interval_sec):
        try:
            store.heartbeat(task_id)
        except Exception:
            logger.exception("Задача %s: не удалось обновить heartbeat", task_id)


def run_task(
    task: TaskState,
    bundle: ModelBundle,
    settings: Settings,
    store: TaskStore,
    cancellation: CancellationRegistry,
    s3_client: S3Client,
    logger: logging.Logger,
) -> None:
    task_id = task.task_id
    processed_rows = 0
    inference_elapsed_sec = 0.0
    last_progress_flush = time.monotonic()
    success_written = False
    heartbeat_stop = threading.Event()
    heartbeat_thread = None
    heartbeat_started = False

    def cancelled() -> bool:
        # Локальный флаг ускоряет отмену на своём поде; S3 доставляет её с других.
        return cancellation.is_cancelled(task_id) or store.cancellation_requested(task_id)

    def abort() -> None:
        store.update_progress(task_id, processed_rows, inference_elapsed_sec)
        store.set_status(task_id, TaskStatus.ABORTED)
        logger.info("Задача %s отменена: processed_rows=%s", task_id, processed_rows)

    try:
        heartbeat_thread = threading.Thread(
            target=_keep_heartbeat,
            args=(task_id, store, heartbeat_stop, logger),
            name=f"task-heartbeat-{task_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        heartbeat_started = True
        logger.info("Задача %s начата: total_rows=%s", task_id, task.total_rows)

        if cancelled():
            abort()
            return

        from app.models.inference import run_inference_on_chunk

        # Ошибка инициализации также должна завершать задачу и освобождать слот.
        writer = s3_client.get_writer(task.s3_output_path)

        for df_chunk in s3_client.iter_chunks(task.s3_input_path, settings.inference.chunk_size):
            if cancelled():
                abort()
                return

            started = time.perf_counter()
            result_chunk = run_inference_on_chunk(
                df_chunk, bundle, settings.inference.infer_batch_size
            )
            inference_elapsed_sec += time.perf_counter() - started

            if cancelled():
                abort()
                return

            writer.write_chunk(result_chunk)
            processed_rows += len(df_chunk)

            if time.monotonic() - last_progress_flush >= settings.task.progress_update_interval_sec:
                store.update_progress(task_id, processed_rows, inference_elapsed_sec)
                last_progress_flush = time.monotonic()

        if cancellation.is_cancelled(task_id):
            abort()
            return

        # Проверка отмены и решение о завершении выполняются в одной мутации S3.
        # После FINALIZING API уже не принимает отмену: пишется маркер успеха.
        if not store.prepare_completion(task_id, processed_rows, inference_elapsed_sec):
            logger.info("Задача %s: завершение отменено", task_id)
            return

        writer.close()
        success_written = True
        # Запись DONE идемпотентна. Ошибка PUT статуса после публикации результата
        # не должна превращать успешно завершённый инференс в FAILED.
        for attempt in range(3):
            try:
                store.set_status(task_id, TaskStatus.DONE)
                break
            except Exception:
                if attempt == 2:
                    raise
                logger.warning("Задача %s: повтор сохранения DONE", task_id, exc_info=True)
                time.sleep(1.0)
        logger.info("Задача %s завершена: processed_rows=%s", task_id, processed_rows)

    except FileNotFoundError as exc:
        logger.exception("Задача %s: путь S3 стал недоступен", task_id)
        if not success_written:
            store.set_status(task_id, TaskStatus.FAILED, error=f"s3_path_not_found: {exc}")
    except Exception as exc:
        if success_written:
            logger.exception("Задача %s: результат опубликован, сохранение DONE не подтверждено", task_id)
        else:
            logger.exception("Задача %s упала с ошибкой", task_id)
            store.set_status(task_id, TaskStatus.FAILED, error=str(exc))
    finally:
        heartbeat_stop.set()
        if heartbeat_started and heartbeat_thread is not None:
            # Вызов S3 может ещё ожидать сетевой timeout. Поток daemon, а TaskStore
            # не разрешает heartbeat менять терминальное состояние задачи.
            heartbeat_thread.join(timeout=1.0)
            if heartbeat_thread.is_alive():
                logger.warning("Задача %s: heartbeat ожидает завершения запроса S3", task_id)
        if not success_written:
            logger.warning(
                "Задача %s: _SUCCESS не записан (частичная или прерванная загрузка)",
                task_id,
            )
        cancellation.cleanup(task_id)

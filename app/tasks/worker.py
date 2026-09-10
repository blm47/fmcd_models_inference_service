"""Фоновый инференс с отменой через S3 и отдельным heartbeat.

Отмена кооперативная: уже начатый чанк инференса завершается, после чего
его результат не записывается, если поступила отмена. Перед записью _SUCCESS
TaskStore фиксирует переход в FINALIZING или принимает последнюю отмену.
Heartbeat показывает доступность исполнителя, а не движение прогресса.
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

from app.tasks.state import TERMINAL_STATUSES, TaskState, TaskStatus, TaskStore

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.models.registry import ModelBundle
    from app.storage.s3_client import S3Client


def _keep_heartbeat(
    task_id: str,
    store: TaskStore,
    stop: threading.Event,
    logger: Any,
    lost: threading.Event | None = None,
) -> None:
    """Не блокирует heartbeat на долгом чтении, forward-pass или записи чанка."""
    while not stop.wait(store.heartbeat_interval_sec):
        try:
            if store.heartbeat(task_id) is False:
                if lost is not None:
                    lost.set()
                return
        except Exception:
            logger.error(
                (f"Задача {task_id}: не удалось обновить heartbeat") + "\n" + traceback.format_exc()
            )


def _save_status(store, task_id, status, error, stop, logger) -> bool:
    """Сохраняет итог до подтверждения S3; повтор не запускает инференс заново."""
    while True:
        try:
            if store.set_status(task_id, status, error=error) is False:
                logger.warn(
                    f"Задача {task_id}: итог {status.value} не принят, уже сохранён другой статус"
                )
                return False
            return True
        except Exception:
            logger.error(
                (f"Задача {task_id}: не удалось сохранить {status.value}; ожидается повтор")
                + "\n"
                + traceback.format_exc()
            )
            if stop.wait(store.config.poll_interval_sec):
                return False


def run_task(
    task: TaskState,
    bundle: ModelBundle,
    settings: Settings,
    store: TaskStore,
    stop: threading.Event,
    s3_client: S3Client,
    logger: Any,
) -> None:
    task_id = task.task_id
    processed_rows = 0
    inference_elapsed_sec = 0.0
    last_progress_flush = time.monotonic()
    success_written = False
    heartbeat_stop = threading.Event()
    heartbeat_thread = None
    heartbeat_started = False
    ownership_lost = threading.Event()

    def check_shutdown() -> None:
        if stop.is_set():
            raise RuntimeError("pod_shutdown: под получил сигнал остановки")
        if ownership_lost.is_set():
            raise RuntimeError("task_not_active: задача завершена по таймауту")

    def cancelled() -> bool:
        check_shutdown()
        return store.cancellation_requested(task_id)

    def abort() -> None:
        store.update_progress(task_id, processed_rows, inference_elapsed_sec)
        if _save_status(store, task_id, TaskStatus.ABORTED, None, stop, logger):
            logger.info(f"Задача {task_id} отменена: processed_rows={processed_rows}")

    try:
        heartbeat_thread = threading.Thread(
            target=_keep_heartbeat,
            args=(task_id, store, heartbeat_stop, logger, ownership_lost),
            name=f"task-heartbeat-{task_id}",
            daemon=True,
        )
        heartbeat_thread.start()
        heartbeat_started = True
        logger.info(f"Задача {task_id} начата: total_rows={task.total_rows}")

        if cancelled():
            abort()
            return

        from app.models.inference import run_inference_on_chunk

        # Ошибка инициализации также должна завершать задачу и освобождать слот.
        if s3_client.prefix_has_results(task.s3_output_path):
            raise ValueError(
                "Выходной префикс не пуст; Airflow должен очистить его перед новым расчётом"
            )
        writer = s3_client.get_writer(task.s3_output_path)

        for df_chunk in s3_client.iter_chunks(task.s3_input_path, settings.inference.chunk_size):
            if cancelled():
                abort()
                return

            started = time.perf_counter()
            result_chunk = run_inference_on_chunk(
                df_chunk, bundle, settings.inference.infer_batch_size, check_shutdown
            )
            inference_elapsed_sec += time.perf_counter() - started

            if cancelled():
                abort()
                return

            writer.write_chunk(result_chunk)
            processed_rows += len(df_chunk)

            if (
                time.monotonic() - last_progress_flush
                >= settings.task_store.progress_update_interval_sec
            ):
                store.update_progress(task_id, processed_rows, inference_elapsed_sec)
                last_progress_flush = time.monotonic()

        check_shutdown()

        # Проверка отмены и решение о завершении выполняются в одной мутации S3.
        # После FINALIZING API уже не принимает отмену: пишется маркер успеха.
        if not store.prepare_completion(task_id, processed_rows, inference_elapsed_sec):
            logger.info(f"Задача {task_id}: завершение отменено или задача просрочена")
            return

        check_shutdown()
        writer.close()
        success_written = True
        # Запись DONE идемпотентна. Ошибка PUT статуса после публикации результата
        # не должна превращать успешно завершённый инференс в FAILED.
        if _save_status(store, task_id, TaskStatus.DONE, None, stop, logger):
            logger.info(f"Задача {task_id} завершена: processed_rows={processed_rows}")

    except FileNotFoundError as exc:
        logger.error((f"Задача {task_id}: путь S3 стал недоступен") + "\n" + traceback.format_exc())
        if not success_written:
            _save_status(
                store, task_id, TaskStatus.FAILED, f"s3_path_not_found: {exc}", stop, logger
            )
    except Exception as exc:
        if success_written:
            logger.error(
                (f"Задача {task_id}: результат опубликован, сохранение DONE не подтверждено")
                + "\n"
                + traceback.format_exc()
            )
        else:
            logger.error((f"Задача {task_id} упала с ошибкой") + "\n" + traceback.format_exc())
            _save_status(store, task_id, TaskStatus.FAILED, str(exc), stop, logger)
    finally:
        heartbeat_stop.set()
        if heartbeat_started and heartbeat_thread is not None:
            # Вызов S3 может ещё ожидать сетевой timeout. Поток daemon, а TaskStore
            # не разрешает heartbeat менять терминальное состояние задачи.
            heartbeat_thread.join(timeout=1.0)
            if heartbeat_thread.is_alive():
                logger.warn(f"Задача {task_id}: heartbeat ожидает завершения запроса S3")
        if not success_written:
            logger.warn(
                f"Задача {task_id}: _SUCCESS не записан (частичная или прерванная загрузка)"
            )


def consume_queue(store, models, settings, s3_client, pod_id, stop, logger) -> None:
    """Один поток последовательно выполняет задачи; занятый GPU не блокирует API."""
    current_task = None
    while not stop.is_set():
        try:
            if current_task is not None:
                task = store.get(current_task.task_id)
                if task is None or task.status not in TERMINAL_STATUSES:
                    # Worker уже вышел, но финальный статус не подтверждён.
                    # Не запускаем тот же инференс повторно и не теряем занятый слот.
                    stop.wait(settings.task_store.poll_interval_sec)
                    continue
                current_task = None
            current_task = store.claim_next(pod_id, set(models))
            if current_task is not None:
                run_task(
                    current_task,
                    models[current_task.model_name],
                    settings,
                    store,
                    stop,
                    s3_client,
                    logger,
                )
                continue
        except Exception:
            logger.error(
                (f"Под {pod_id}: ошибка обработки очереди; следующая проверка повторит чтение")
                + "\n"
                + traceback.format_exc()
            )
        stop.wait(settings.task_store.poll_interval_sec)


def monitor_queue(store, stop, logger) -> None:
    """Проверяет таймауты независимо от занятости GPU на этом поде."""
    next_cleanup = time.monotonic()
    while not stop.is_set():
        try:
            store.fail_stale()
            if time.monotonic() >= next_cleanup:
                store.cleanup()
                next_cleanup = time.monotonic() + store.config.cleanup_interval_sec
        except Exception:
            logger.error(
                ("Не удалось проверить таймауты очереди S3") + "\n" + traceback.format_exc()
            )
        stop.wait(store.config.poll_interval_sec)

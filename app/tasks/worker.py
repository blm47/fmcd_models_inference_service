"""
Фоновый инференс с отменой и контролем активности через TaskStore.

Отмена кооперативная: уже начатый чанк инференса завершается, после чего
его результат не записывается, если поступила отмена. Перед записью _SUCCESS
TaskStore фиксирует переход в FINALIZING или принимает последнюю отмену.
Heartbeat обновляет last_modified независимо от длительности этапов инференса.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any

from app.models.contracts import column_names
from app.models.registry import create_bundle
from app.tasks.state import TERMINAL_STATUSES, TaskState, TaskStatus, TaskStore
from app.tasks.utilization import UtilizationSampler

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.models.contracts import ModelSpec
    from app.storage.s3_client import S3Client


def _keep_heartbeat(task_id, store, stop, lost, logger) -> None:
    """
    Обновляет last_modified независимо от load, infer и записи результата.
    """
    while not stop.wait(store.config.heartbeat_interval_sec):
        try:
            if not store.heartbeat(task_id):
                lost.set()
                return
        except Exception:
            logger.error(
                f"Задача {task_id}: не удалось обновить heartbeat\n" + traceback.format_exc()
            )


def _save_status(store, task_id, status, error, stop, logger) -> bool:
    """
    Сохраняет итог до подтверждения хранилища; повтор не запускает инференс заново.
    """
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


class TaskCancelled(Exception):
    """
    Кооперативная отмена между стадиями и infer-батчами.
    """


def run_task(
    task: TaskState,
    spec: ModelSpec,
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
    bundle = None
    df_chunk = result_chunk = chunks = None
    output_dtypes = None
    load_complete = False
    utilization = None
    heartbeat_stop = threading.Event()
    ownership_lost = threading.Event()
    heartbeat_thread = None
    heartbeat_started = False

    def check_shutdown() -> None:
        if stop.is_set():
            raise RuntimeError("pod_shutdown: под получил сигнал остановки")
        if ownership_lost.is_set():
            raise RuntimeError("task_not_active: задача завершена или просрочена")

    def cancelled() -> bool:
        check_shutdown()
        return store.cancellation_requested(task_id)

    def abort() -> None:
        store.update_progress(task_id, processed_rows, inference_elapsed_sec)
        if _save_status(store, task_id, TaskStatus.ABORTED, None, stop, logger):
            logger.info(f"Задача {task_id} отменена: processed_rows={processed_rows}")

    def check_pipeline() -> None:
        if cancelled():
            raise TaskCancelled()

    try:
        heartbeat_thread = threading.Thread(
            target=_keep_heartbeat,
            args=(task_id, store, heartbeat_stop, ownership_lost, logger),
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
        from app.models.validation import validate_input_parquet

        if task.calc_utilization:
            utilization = UtilizationSampler(task_id, spec.device, logger)
            utilization.start()
        bundle = create_bundle(spec, logger)
        logger.info(f"Задача {task_id}, модель={spec.name}: начало load")
        bundle.load()
        load_complete = True
        logger.info(f"Задача {task_id}, модель={spec.name}: load завершён")
        check_pipeline()
        bundle.required_columns = column_names(
            bundle.required_columns, "required_columns", allow_empty=True
        )
        bundle.output_columns = column_names(bundle.output_columns, "output_columns")
        if set(spec.id_cols) & set(bundle.output_columns):
            raise ValueError("Выходы pipeline не должны дублировать id_cols")
        validation = validate_input_parquet(task.s3_input_path, bundle, s3_client)
        if not validation.is_valid:
            raise ValueError(f"missing_columns: {validation.missing_columns}")
        if cancelled():
            abort()
            return
        if store.set_total_rows(task_id, validation.total_rows) is False:
            if cancelled():
                abort()
            return
        task.total_rows = validation.total_rows
        logger.info(f"Задача {task_id}: вход проверен, total_rows={task.total_rows}")

        # Ошибка инициализации также должна завершать задачу и освобождать слот.
        if s3_client.prefix_has_results(task.s3_output_path):
            raise ValueError(
                "Выходной префикс не пуст; Airflow должен очистить его перед новым расчётом"
            )
        writer = s3_client.get_writer(task.s3_output_path)

        chunks = iter(s3_client.iter_chunks(task.s3_input_path, spec.parquet_read_chunk_size))
        while True:
            check_pipeline()
            df_chunk = next(chunks, None)
            if df_chunk is None:
                break
            if cancelled():
                abort()
                return

            if len(df_chunk) == 0:
                continue

            started = time.perf_counter()
            result_chunk = run_inference_on_chunk(
                df_chunk, bundle, spec.infer_batch_size, check_pipeline
            )
            inference_elapsed_sec += time.perf_counter() - started
            current_dtypes = tuple(result_chunk.dtypes)
            if output_dtypes is not None and current_dtypes != output_dtypes:
                raise ValueError("Типы результата изменились между чанками")
            output_dtypes = current_dtypes

            if cancelled():
                abort()
                return

            writer.write_chunk(result_chunk)
            processed_rows += len(df_chunk)
            df_chunk = result_chunk = None

            if (
                time.monotonic() - last_progress_flush
                >= settings.task_store.progress_update_interval_sec
            ):
                store.update_progress(task_id, processed_rows, inference_elapsed_sec)
                last_progress_flush = time.monotonic()

        check_shutdown()

        if processed_rows != task.total_rows:
            raise ValueError("Количество прочитанных строк изменилось после проверки входа")

        # Проверка отмены и решение о завершении выполняются в одной мутации очереди.
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

    except TaskCancelled:
        abort()
    except FileNotFoundError as exc:
        reason = "s3_path_not_found" if load_complete else "model_artifact_not_found"
        logger.error(f"Задача {task_id}: {reason}\n" + traceback.format_exc())
        if not success_written:
            _save_status(store, task_id, TaskStatus.FAILED, f"{reason}: {exc}", stop, logger)
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
        df_chunk = result_chunk = chunks = None
        if bundle is not None:
            try:
                logger.info(f"Задача {task_id}, модель={spec.name}: начало close")
                try:
                    bundle.close()
                finally:
                    bundle = None
                logger.info(f"Задача {task_id}, модель={spec.name}: close завершён")
            except Exception:
                stop.set()
                logger.error(
                    f"Задача {task_id}, модель={spec.name}: ошибка close; под остановлен\n"
                    + traceback.format_exc()
                )
            finally:
                bundle = None
        if utilization is not None:
            try:
                utilization.stop()
                logger.info(
                    f"Задача {task_id}: итоговая утилизация "
                    + json.dumps(utilization.snapshot(), ensure_ascii=False, allow_nan=False)
                )
            except Exception:
                logger.error(
                    f"Задача {task_id}: не удалось завершить сбор утилизации\n"
                    + traceback.format_exc()
                )
        heartbeat_stop.set()
        if heartbeat_started:
            heartbeat_thread.join(timeout=1.0)
            if heartbeat_thread.is_alive():
                logger.warn(f"Задача {task_id}: heartbeat ожидает завершения запроса к хранилищу")
        if not success_written:
            logger.warn(
                f"Задача {task_id}: _SUCCESS не записан (частичная или прерванная загрузка)"
            )


def consume_queue(store, models, settings, s3_client, pod_id, stop, logger) -> None:
    """
    Единственный и главный поток воркера
    последовательно выполняет задачи; занятый GPU не блокирует API.
    """
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

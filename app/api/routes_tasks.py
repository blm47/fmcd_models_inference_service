"""
GET /tasks/{task_id}/status и POST /tasks/{task_id}/abort.

Статусы и запросы отмены разделяются между подами через S3. История
хранится до retention_months; незавершённые задачи не удаляются.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
    ActiveTasksResponse,
    ActiveTaskSummary,
    ErrorResponse,
    TaskAbortResponse,
    TaskStatusResponse,
)
from app.deps import get_task_store
from app.tasks.state import TaskStore

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    responses={
        503: {"model": ErrorResponse, "description": "Не удалось прочитать или обновить очередь S3"}
    },
)


@router.get(
    "/{task_id}/status",
    response_model=TaskStatusResponse,
    summary="Получить статус и прогресс расчёта",
    responses={
        404: {"model": ErrorResponse, "description": "Задача не найдена или удалена из истории"}
    },
)
def get_task_status(
    task_id: str,
    task_store: TaskStore = Depends(get_task_store),
):
    """
    Читает общую очередь, поэтому запрос можно направить на любой под.

    Airflow опрашивает этот метод до DONE, FAILED или ABORTED.
    Результат готов к чтению только при DONE. ETA является оценкой, а не дедлайном.
    `executor_alive` вычисляется по heartbeat и не подтверждает физическую
    остановку процесса при значении false.
    """
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача {task_id} не найдена")

    return TaskStatusResponse(
        task_id=task.task_id,
        model_name=task.model_name,
        pod_id=task.pod_id,
        status=task.status,
        processed_rows=task.processed_rows,
        total_rows=task.total_rows,
        progress_pct=task.progress_pct,
        eta_seconds=task.eta_seconds,
        error=task.error,
        heartbeat_at=task.heartbeat_at,
        executor_alive=task_store.executor_alive(task),
        s3_output_path=task.s3_output_path,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


@router.post(
    "/{task_id}/abort",
    response_model=TaskAbortResponse,
    summary="Запросить отмену расчёта",
    responses={
        404: {"model": ErrorResponse, "description": "Задача не найдена"},
        409: {"model": ErrorResponse, "description": "Задача FINALIZING уже публикует результат"},
    },
)
def abort_task(task_id: str, task_store: TaskStore = Depends(get_task_store)):
    """
    QUEUED отменяется сразу, RUNNING переходит в ABORTING.

    Worker завершает текущую операцию и подтверждает ABORTED при проверке отмены.
    Ответ ABORTING ещё не означает остановку. Для завершённой задачи возвращается
    её текущий статус. Этот метод не удаляет выходные файлы и не запускает повтор.
    """
    task = task_store.request_abort(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача {task_id} не найдена")

    return TaskAbortResponse(task_id=task.task_id, status=task.status)


@router.get(
    "/active", response_model=ActiveTasksResponse, summary="Получить все незавершённые задачи"
)
def get_active_tasks(task_store: TaskStore = Depends(get_task_store)):
    """
    Все незавершённые задачи, включая очередь и потерявших heartbeat исполнителей.
    """
    active_tasks = task_store.get_all_active()
    return ActiveTasksResponse(
        active_tasks=[
            ActiveTaskSummary(
                task_id=task.task_id,
                pod_id=task.pod_id,
                model_name=task.model_name,
                status=task.status,
                progress_pct=task.progress_pct,
                eta_seconds=task.eta_seconds,
                executor_alive=task_store.executor_alive(task),
            )
            for task in active_tasks
        ]
    )

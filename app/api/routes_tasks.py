"""
GET /tasks/{task_id}/status и POST /tasks/{task_id}/abort.

Статусы и запросы отмены разделяются между подами через S3. История
хранится до retention_months; незавершённые задачи не удаляются.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
    ActiveTasksResponse,
    ActiveTaskSummary,
    TaskAbortResponse,
    TaskStatusResponse,
)
from app.deps import get_task_store
from app.tasks.state import TaskStore

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
def get_task_status(
    task_id: str,
    task_store: TaskStore = Depends(get_task_store),
):
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


@router.post("/{task_id}/abort", response_model=TaskAbortResponse)
def abort_task(task_id: str, task_store: TaskStore = Depends(get_task_store)):
    task = task_store.request_abort(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача {task_id} не найдена")

    return TaskAbortResponse(task_id=task.task_id, status=task.status)


@router.get("/active", response_model=ActiveTasksResponse)
def get_active_tasks(task_store: TaskStore = Depends(get_task_store)):
    """Все незавершённые задачи, включая очередь и потерявших heartbeat исполнителей."""
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

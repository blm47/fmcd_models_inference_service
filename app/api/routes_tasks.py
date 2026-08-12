"""
GET /tasks/{task_id}/status и POST /tasks/{task_id}/abort.

Читают/мутируют состояние из in-memory TaskStore/CancellationRegistry —
по решению архитектуры история задач хранится "в рамках жизни пода" без
внешнего стораджа, поэтому 404 отдаётся только если task_id вообще не
встречался за время жизни текущего пода.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import TaskAbortResponse, TaskStatusResponse
from app.deps import get_task_manager
from app.tasks.manager import TaskManager

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
def get_task_status(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    task = task_manager.get_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача {task_id} не найдена")

    return TaskStatusResponse(
        task_id=task.task_id,
        model_name=task.model_name,
        status=task.status,
        processed_rows=task.processed_rows,
        total_rows=task.total_rows,
        progress_pct=task.progress_pct,
        eta_seconds=task.eta_seconds,
        error=task.error,
    )


@router.post("/{task_id}/abort", response_model=TaskAbortResponse)
def abort_task(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    task = task_manager.request_abort(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача {task_id} не найдена")

    return TaskAbortResponse(task_id=task.task_id, status=task.status)

"""
GET /tasks/{task_id}/status и POST /tasks/{task_id}/abort.

Читают/мутируют состояние из in-memory TaskStore/CancellationRegistry —
по решению архитектуры история задач хранится "в рамках жизни пода" без
внешнего стораджа, поэтому 404 отдаётся только если task_id вообще не
встречался за время жизни текущего пода.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
    ActiveTaskSummary,
    ActiveTasksResponse,
    TaskAbortResponse,
    TaskStatusResponse,
)
from app.deps import get_task_manager, get_task_store
from app.tasks.manager import TaskManager
from app.tasks.state import TaskStore

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}/status", response_model=TaskStatusResponse)
def get_task_status(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    task = task_manager.get_status(task_id)
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
    )


@router.post("/{task_id}/abort", response_model=TaskAbortResponse)
def abort_task(task_id: str, task_manager: TaskManager = Depends(get_task_manager)):
    task = task_manager.request_abort(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Задача {task_id} не найдена")

    return TaskAbortResponse(task_id=task.task_id, status=task.status)


@router.get("/active", response_model=ActiveTasksResponse)
def get_active_tasks(task_store: TaskStore = Depends(get_task_store)):
    """
    Диагностический эндпоинт: список ВСЕХ активных задач по всем подам
    сервиса. При правиле "1 активная задача на 1 под" и N подах здесь
    может быть до N записей одновременно - удобно для мониторинга/Airflow,
    чтобы видеть загрузку сервиса в целом, а не только одного пода
    (на который случайно попал запрос через балансировщик).
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
            )
            for task in active_tasks
        ]
    )

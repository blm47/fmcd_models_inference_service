"""
Dependency Injection для FastAPI-роутов.

Все "тяжёлые" объекты (dict моделей, TaskStore, TaskManager, S3Client)
создаются один раз в app.main:lifespan и кладутся в app.state, а роуты
достают их через Depends(...).
"""

from fastapi import Request

from app.core.config import Settings, load_settings
from app.models.registry import ModelBundle
from app.storage.s3_client import S3Client
from app.tasks.cancellation import CancellationRegistry
from app.tasks.manager import TaskManager
from app.tasks.state import TaskStore


def get_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or load_settings()


def get_models(request: Request) -> dict[str, ModelBundle]:
    """dict[model_name -> ModelBundle], наполняется один раз в lifespan."""
    return request.app.state.models


def get_task_store(request: Request) -> TaskStore:
    return request.app.state.task_store


def get_cancellation_registry(request: Request) -> CancellationRegistry:
    return request.app.state.cancellation_registry


def get_task_manager(request: Request) -> TaskManager:
    return request.app.state.task_manager


def get_s3_client(request: Request) -> S3Client:
    return request.app.state.s3_client

"""Объекты из lifespan передаются в API без повторной инициализации."""

from fastapi import Request


def get_settings(request: Request):
    return request.app.state.settings


def get_models(request: Request):
    return request.app.state.models


def get_task_store(request: Request):
    return request.app.state.task_store


def get_s3_client(request: Request):
    return request.app.state.s3_client


def get_logger(request: Request):
    return request.app.state.logger

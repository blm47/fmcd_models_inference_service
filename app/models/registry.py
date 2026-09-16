"""
Ленивый реестр backend: каталог не импортирует библиотеки моделей.
"""

from importlib import import_module
from typing import Any

from app.models.contracts import ModelBundle, ModelSpec

BACKENDS = {
    "fmcd_cc_dc": "app.models.backends.fmcd_cc_dc.runner:FMCDCardsRunner",
    "fmcd_invest": "app.models.backends.fmcd_invest.runner:FMCDInvestRunner",
}


def validate_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"Неизвестный backend: {backend}")


def create_bundle(spec: ModelSpec, logger: Any) -> ModelBundle:
    """
    Создаёт новый экземпляр для задачи, не загружая веса в фабрике.
    """
    validate_backend(spec.backend)
    module_name, class_name = BACKENDS[spec.backend].split(":")
    cls = getattr(import_module(module_name), class_name)
    if not isinstance(cls, type) or not issubclass(cls, ModelBundle):
        raise TypeError(f"Backend {spec.backend} должен наследовать ModelBundle")
    return cls(spec, logger)

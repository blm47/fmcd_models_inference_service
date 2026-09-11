"""
ModelBundle - всё, что нужно для инференса ОДНОЙ модели: веса + препроцессинг-
артефакты. Сам реестр моделей - обычный dict[str, ModelBundle] (см.
app/main.py: app.state.models), отдельный класс-обёртка не нужен, т.к.
логика сводится к чтению по ключу без дополнительного поведения.
"""

from dataclasses import dataclass
from typing import Any

import torch
from fmcd.data.production import ProdSchema


@dataclass
class ModelBundle:
    """
    Всё, что нужно для инференса ОДНОЙ модели: веса + препроцессинг-артефакты.
    """

    name: str
    model: Any  # torch.nn.Module (FMCDModel), тип из fmcd.model.fmcd_model
    schema: ProdSchema
    calibrators: dict  # Коэффициенты Platt calibration для каждого MCG.
    device: torch.device

    id_cols: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    d_cols: list[str]
    f_cols: list[str]
    m_cols: list[str]
    c_cols: list[str]

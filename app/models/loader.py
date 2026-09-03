"""
Загрузка модели и её препроцессинг-артефактов в ModelBundle.

Логика 1-в-1 повторяет ячейки из ноутбука моделистов:
  - torch.serialization.add_safe_globals([FMCDModel])
  - torch.load(model_path, map_location=device, weights_only=False)
  - model.cat_processor.set_frequency_encoding(...) из freq_counts
  - ProdSchema.from_json_local(schema.json)
  - calibrators.json -> calibs_dict

Вызывается для каждой модели из config/models.yaml -> models[] один раз
в lifespan (app/main.py): на старте пода - веса лежат в образе, артефакты
лежат в artifacts/<model_name>/, поэтому холодный старт пода - это и есть
точка загрузки, без ленивой подгрузки по запросу.
"""

import logging
import json

import numpy as np
import torch
import torch.serialization

from fmcd.data.production import ProdSchema, load_freq_counts_local
from fmcd.model.fmcd_model import FMCDModel

from app.core.config import ModelConfig, InferenceConfig
from app.models.registry import ModelBundle

# logger = logging.getLogger(__name__)


def load_model_bundle(model_cfg: ModelConfig, inference_cfg: InferenceConfig, logger: logging.Logger) -> ModelBundle:
    device = torch.device(inference_cfg.device if torch.cuda.is_available() else "cpu")
    if inference_cfg.device == "cuda" and device.type != "cuda":
        logger.warn("CUDA запрошена в конфиге, но недоступна - падаем на CPU")

    schema = ProdSchema.from_json_local(model_cfg.schema_path)

    d_cols = schema.d_cols or [f"d_{k}" for k in range(schema.num_mcg)]
    f_cols = schema.f_cols or [f"f_{k}" for k in range(schema.num_mcg)]
    m_cols = schema.m_cols or [f"m_{k}" for k in range(schema.num_mcg)]
    c_cols = schema.c_cols or [f"c_{k}" for k in range(schema.num_mcg)]
    num_cols = schema.num_cols or [f"num_{i}" for i in range(schema.num_numerical)]
    cat_cols = schema.cat_cols or [f"cat_{i}" for i in range(schema.num_categorical)]

    torch.serialization.add_safe_globals([FMCDModel])
    model = torch.load(model_cfg.weights_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    freq_counts = load_freq_counts_local(model_cfg.freq_encoding_path, schema)
    for i, counts in enumerate(freq_counts):
        model.cat_processor.set_frequency_encoding(i, torch.from_numpy(counts))

    with open(model_cfg.calibrators_path, "r", encoding="utf-8") as f:
        calibrators = json.loads(f.read())

    id_cols = model_cfg.id_cols

    calib_intercepts = np.array(
        [calibrators[col]["intercept"] for col in d_cols],
        dtype=np.float64,
    )  # shape (n_d_cols,)
    calib_coefs = np.array(
        [calibrators[col]["coef"] for col in d_cols],
        dtype=np.float64,
    )  # shape (n_d_cols,)

    logger.info(
        f"Модель '{model_cfg.name}' загружена: device={device}, MCG={schema.num_mcg}, "
        f"num_features={schema.num_numerical}, cat_features={schema.num_categorical}",
    )
    return ModelBundle(
        name=model_cfg.name,
        model=model,
        schema=schema,
        calibrators=calibrators,
        device=device,
        id_cols=id_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        d_cols=d_cols,
        f_cols=f_cols,
        m_cols=m_cols,
        c_cols=c_cols,
        calib_intercepts=calib_intercepts,
        calib_coefs=calib_coefs,
    )


def load_all_models(
        model_cfgs: list[ModelConfig],
        inference_cfg: InferenceConfig,
        logger: logging.Logger
) -> dict[str, ModelBundle]:
    """Загружает все модели из конфига в dict[name -> ModelBundle] для app.state.models."""
    return {cfg.name: load_model_bundle(cfg, inference_cfg, logger) for cfg in model_cfgs}

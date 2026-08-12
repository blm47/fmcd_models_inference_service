"""
Прямой перенос функции pandas_chunk_to_fmcd_batch из вашего ноутбука без изменений
логики. Вынесена в отдельный модуль, чтобы models/inference.py не разрастался
и чтобы эту функцию было легко покрыть unit-тестом изолированно (сравнение
с ожидаемыми shape'ами тензоров на синтетическом df_chunk).
"""

import numpy as np
import pandas as pd
import torch

from fmcd.data.base import FMCDBatch
from fmcd.data.production import ProdSchema


def pandas_chunk_to_fmcd_batch(
    df_chunk: pd.DataFrame,
    schema: ProdSchema,
    num_cols: list[str],
    cat_cols: list[str],
) -> FMCDBatch:
    """
    Конвертирует pandas DataFrame -> FMCDBatch.
    Логика идентична HDFSFMCDDataset._process_batch, но для pandas.
    Таргеты заполняются нулями (не нужны на инференсе).
    """
    B = len(df_chunk)

    num_arrays, mask_arrays = [], []
    for col in num_cols:
        vals = df_chunk[col].values
        mask_arrays.append((~pd.isna(df_chunk[col])).astype(np.float32))
        num_arrays.append(np.nan_to_num(vals, nan=0.0).astype(np.float32))

    num = np.stack(num_arrays, axis=1)
    mask = np.stack(mask_arrays, axis=1)

    cat_arrays = []
    for i, col in enumerate(cat_cols):
        cardinality = schema.cat_cardinalities[i]
        idx = df_chunk[col].fillna(cardinality).astype(np.int64).values
        idx = np.clip(idx, 0, cardinality)
        cat_arrays.append(idx)
    cat = np.stack(cat_arrays, axis=1)

    return FMCDBatch(
        num_features=torch.from_numpy(num),
        cat_features=torch.from_numpy(cat),
        missing_mask=torch.from_numpy(mask),
        d_multilabel=torch.zeros(B, schema.num_mcg),
        d_count=torch.zeros(B, 1),
        f_targets=torch.zeros(B, schema.num_mcg),
        m_targets=torch.zeros(B, schema.num_mcg),
        c_targets=torch.zeros(B, schema.num_mcg),
    )

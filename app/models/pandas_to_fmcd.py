"""
Прямой перенос функции pandas_chunk_to_fmcd_batch из ноутбука моделистов без изменений
логики. Вынесена в отдельный модуль, чтобы models/inference.py не разрастался.
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

    block = df_chunk[num_cols].to_numpy(dtype=np.float32, copy=True)
    mask = (~np.isnan(block)).astype(np.float32)
    num = np.nan_to_num(block, nan=0.0)

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

"""
Батчевый инференс одного чанка данных — прямой перенос логики из ноутбука:

  pandas_chunk_to_fmcd_batch -> model(fmcd_batch) -> sigmoid(d_logits) ->
  Platt calibration (log_reg + _logit) -> expm1-денормализация F/M/C/D_count.

Отличие от ноутбука: здесь функция работает над ОДНИМ чанком (для потоковой
S3-записи), а разбиение чанка на INFER_BATCH_SIZE под-батчи для forward-pass
остаётся внутренним делом этой функции (GPU H100 может считать батчами
побольше, чем в ноутбучном примере на CPU).
"""

import numpy as np
import pandas as pd
import torch

from app.models.registry import ModelBundle

ID_COL = "customer_mdm_id"
DATE_COL = "partition_report_dt"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _log_reg(intercept: float, coef: float, x: np.ndarray) -> np.ndarray:
    z = intercept + coef * x
    return 1.0 / (1.0 + np.exp(-z))


def run_inference_on_chunk(
    df_chunk: pd.DataFrame,
    bundle: ModelBundle,
    infer_batch_size: int,
) -> pd.DataFrame:
    """
    Возвращает DataFrame с результатами (ID_COL, DATE_COL + d_prob_*/f_pred_*/
    m_pred_*/c_pred_*/d_count_pred) для одного входного чанка, разбивая его
    на под-батчи размера infer_batch_size для forward-pass модели.
    """
    from app.models.pandas_to_fmcd import pandas_chunk_to_fmcd_batch  # локальный импорт, см. ниже

    all_results = []
    with torch.no_grad():
        for start in range(0, len(df_chunk), infer_batch_size):
            sub = df_chunk.iloc[start:start + infer_batch_size]
            ids = sub[ID_COL].values
            dates = sub[DATE_COL].values

            fmcd_batch = pandas_chunk_to_fmcd_batch(
                sub, bundle.schema, bundle.num_cols, bundle.cat_cols
            ).to(bundle.device)

            out = bundle.model(fmcd_batch)
            result = {ID_COL: ids, DATE_COL: dates}

            d_probs_raw = torch.sigmoid(out.d_multilabel_logits).cpu().float().numpy()
            for k, col in enumerate(bundle.d_cols):
                calib = bundle.calibrators[col]
                d_prob_calib = _log_reg(
                    calib["intercept"], calib["coef"],
                    np.array(_logit(d_probs_raw[:, k])).reshape(-1, 1)
                ).reshape(1, -1)[0]
                result[f"d_prob_{col}"] = d_prob_calib

            f_log = out.f_value_pred.cpu().float().numpy()
            for k, (f_col, d_col) in enumerate(zip(bundle.f_cols, bundle.d_cols)):
                result[f"f_pred_{f_col}"] = result[f"d_prob_{d_col}"] * np.expm1(f_log[:, k])

            m_log = out.m_value_pred.cpu().float().numpy()
            for k, (m_col, d_col) in enumerate(zip(bundle.m_cols, bundle.d_cols)):
                result[f"m_pred_{m_col}"] = result[f"d_prob_{d_col}"] * np.expm1(m_log[:, k])

            c_log = out.c_value_pred.cpu().float().numpy()
            for k, (c_col, d_col) in enumerate(zip(bundle.c_cols, bundle.d_cols)):
                result[f"c_pred_{c_col}"] = result[f"d_prob_{d_col}"] * np.expm1(c_log[:, k])

            result["d_count_pred"] = np.expm1(out.d_count_pred.cpu().float().numpy().squeeze(1))

            all_results.append(pd.DataFrame(result))

    return pd.concat(all_results, ignore_index=True)

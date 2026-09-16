"""
Вычисления одного legacy-батча: исходные формулы калибровки и денормализации.
"""

import numpy as np
import pandas as pd
import torch


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def log_reg(intercept: float, coef: float, x: np.ndarray) -> np.ndarray:
    z = intercept + coef * x
    return 1.0 / (1.0 + np.exp(-z))


def predict_fmcd_batch(sub, bundle, check_shutdown):
    """
    Выполняет один батч без общего цикла разбиения и очистки CUDA cache.
    """
    from app.models.backends.fmcd_cc_dc.utils.pandas_to_fmcd import pandas_chunk_to_fmcd_batch

    check_shutdown()
    with torch.no_grad():
        fmcd_batch = pandas_chunk_to_fmcd_batch(
            sub, bundle.schema, bundle.num_cols, bundle.cat_cols
        ).to(bundle.device)
        check_shutdown()
        out = bundle.model(fmcd_batch)
        check_shutdown()
        result = {id_col_name: sub[id_col_name].values for id_col_name in bundle.id_cols}

        # D бинарные вероятности - с Platt calibration
        d_probs_raw = torch.sigmoid(out.d_multilabel_logits).cpu().float().numpy()
        for k, col in enumerate(bundle.d_cols):
            calib = bundle.calibrators[col]
            d_prob_calib = log_reg(
                calib["intercept"],
                calib["coef"],
                np.array(logit(d_probs_raw[:, k])).reshape(-1, 1),
            ).reshape(1, -1)[0]
            result[f"d_prob_{col}"] = d_prob_calib

        # F - безусловное: d_prob_calib * expm1(f_log)
        f_log = out.f_value_pred.cpu().float().numpy()
        for k, (f_col, d_col) in enumerate(zip(bundle.f_cols, bundle.d_cols, strict=True)):
            result[f"f_pred_{f_col}"] = result[f"d_prob_{d_col}"] * np.expm1(f_log[:, k])

        # M - безусловное
        m_log = out.m_value_pred.cpu().float().numpy()
        for k, (m_col, d_col) in enumerate(zip(bundle.m_cols, bundle.d_cols, strict=True)):
            result[f"m_pred_{m_col}"] = result[f"d_prob_{d_col}"] * np.expm1(m_log[:, k])

        # C - безусловное
        c_log = out.c_value_pred.cpu().float().numpy()
        for k, (c_col, d_col) in enumerate(zip(bundle.c_cols, bundle.d_cols, strict=True)):
            result[f"c_pred_{c_col}"] = result[f"d_prob_{d_col}"] * np.expm1(c_log[:, k])

        result["d_count_pred"] = np.expm1(out.d_count_pred.cpu().float().numpy().squeeze(1))

    check_shutdown()
    return pd.DataFrame(result)

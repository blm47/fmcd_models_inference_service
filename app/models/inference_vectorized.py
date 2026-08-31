"""
Векторизованная замена d_bin_probe-блока в inference.py.

ПРОБЛЕМА В ИСХОДНОМ КОДЕ:
    for k, col in enumerate(bundle.d_cols):        # 52 колонки
        calib = bundle.calibrators[col]
        d_prob_calib = log_reg(
            calib["intercept"], calib["coef"],
            np.array(logit(d_probs_raw[:, k])).reshape(-1, 1)
        ).reshape(1, -1)[0]
        result[f"d_prob_{col}"] = d_prob_calib

Это Python-цикл на 52 итерации ВНУТРИ цикла по GPU-подбатчам (625 раз при
infer_batch_size=16 на чанк 10_000 строк) = 32_500 вызовов numpy на
микро-массивах. При infer_batch_size=8192 итераций GPU-цикла станет ~2,
но сам d_bin_probe всё равно останется Python-циклом по 52 колонкам -
его тоже стоит векторизовать, т.к. Platt calibration - это простая
линейная операция, полностью выражаемая матричным умножением.

РЕШЕНИЕ: заранее (один раз при загрузке модели, не в inference-цикле!)
собрать intercept и coef всех калибраторов в единые numpy-векторы
формы (n_d_cols,). Тогда calibration для ВСЕХ колонок и ВСЕХ строк
батча считается ОДНИМ векторизованным выражением без Python-цикла.

Platt scaling: p_calib = sigmoid(intercept + coef * logit(p_raw))
Т.к. sigmoid(logit(p)) = p, при coef=1 это тождество, но coef обычно
!= 1, поэтому: p_calib = sigmoid(intercept + coef * logit(p_raw))
"""

import numpy as np


def build_calibration_matrix(bundle) -> tuple[np.ndarray, np.ndarray]:
    """
    Вызывается ОДИН РАЗ при загрузке модели (в loader.py), не в inference.
    Собирает intercept/coef всех d_cols калибраторов в векторы формы
    (n_d_cols,) в ТОМ ЖЕ порядке, что и bundle.d_cols / out.d_multilabel_logits.
    """
    intercepts = np.array(
        [bundle.calibrators[col]["intercept"] for col in bundle.d_cols],
        dtype=np.float64,
    )  # shape (n_d_cols,)
    coefs = np.array(
        [bundle.calibrators[col]["coef"] for col in bundle.d_cols],
        dtype=np.float64,
    )  # shape (n_d_cols,)
    return intercepts, coefs


def logit_stable(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def vectorized_platt_calibration(
    d_probs_raw: np.ndarray,       # shape (batch, n_d_cols) - из torch.sigmoid(...).cpu().numpy()
    calib_intercepts: np.ndarray,  # shape (n_d_cols,) - из build_calibration_matrix, один раз
    calib_coefs: np.ndarray,       # shape (n_d_cols,)
) -> np.ndarray:
    """
    Заменяет весь Python-цикл 'for k, col in enumerate(bundle.d_cols)'
    ОДНИМ векторизованным выражением над всей матрицей (batch, n_d_cols)
    сразу для ВСЕХ строк и ВСЕХ колонок одновременно - без Python-циклов.
    """
    logits_raw = logit_stable(d_probs_raw)                     # (batch, n_d_cols)
    calibrated_logits = calib_intercepts[None, :] + calib_coefs[None, :] * logits_raw
    return 1.0 / (1.0 + np.exp(-calibrated_logits))            # sigmoid, (batch, n_d_cols)


# ==== КАК ЭТО ВЫГЛЯДИТ В inference.py ПОСЛЕ ПРАВКИ ====
#
# В loader.py, один раз при загрузке модели:
#   bundle.calib_intercepts, bundle.calib_coefs = build_calibration_matrix(bundle)
#
# В inference.py, внутри цикла по GPU-подбатчам, БЕЗ Python-цикла по колонкам:
#
#   with stage_timer.track('d_bin_probe'):
#       d_probs_raw = torch.sigmoid(out.d_multilabel_logits).cpu().float().numpy()  # (batch, n_d_cols)
#       d_probs_calib = vectorized_platt_calibration(
#           d_probs_raw, bundle.calib_intercepts, bundle.calib_coefs
#       )  # (batch, n_d_cols) - ОДНА векторная операция вместо 52 numpy-вызовов
#       for k, col in enumerate(bundle.d_cols):
#           result[f"d_prob_{col}"] = d_probs_calib[:, k]   # это уже дёшево - просто срез

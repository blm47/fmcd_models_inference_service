"""
Расчёт обеих веток INVEST и итогового результата без изменения входного батча.
"""

import numpy as np

from .preprocessing import prepare_data_for_score


def combine_scores(result, probabilities, regression, calibrators):
    """
    Сохраняет формулы калибровки, clipping и NumPy rounding из ноутбука.
    """
    if probabilities.shape != (len(result), 3):
        raise ValueError("CatBoost должен вернуть три вероятности на строку: FC/M/D")
    for target, index in zip("fmcd", (0, 1, 0, 2), strict=True):
        result[f"{target}_prob_score"] = probabilities[:, index]
    for target in "fmcd":
        probability = result[f"{target}_prob_score"].to_numpy()
        weights = calibrators[target]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            log_odds = np.log(probability / (1 - probability))
            calibrated = 1 / (1 + np.exp(-(float(weights["w"]) * log_odds + float(weights["w0"]))))
        result[f"{target}_prob_score_calib"] = calibrated
    for target in "fmcd":
        scores = np.asarray(regression[target.upper()], dtype=np.float64)
        if target == "f":
            scores = np.clip(scores, 0.0, None)
        elif target == "c":
            scores = np.clip(scores, 0.0, 62.0)
        elif target == "d":
            scores = np.clip(scores, 0.0, 13.0)
        result[f"{target}_pred"] = np.round(
            result[f"{target}_prob_score_calib"].to_numpy() * scores,
            2 if target == "m" else 0,
        )
    return result


def predict_invest_batch(frame, bundle, check_shutdown):
    import torch

    keys = list(bundle.spec.id_cols)
    missing = set(keys + list(bundle.required_columns)) - set(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют входные колонки INVEST: {sorted(missing)[:10]}")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Ключи INVEST должны быть уникальными и без NULL")
    result = frame.loc[:, keys].copy()
    if frame.empty:
        for col in bundle.output_columns:
            result[col] = np.empty(0, dtype=np.float64)
        return result
    check_shutdown()
    prepared = prepare_data_for_score(
        frame.loc[:, list(bundle.required_columns)],
        bundle.numeric_columns,
        bundle.categorical_columns,
        bundle.fill_columns,
    )
    classifier_frame = prepared.loc[:, bundle.classifier_columns].copy()
    for col in bundle.classifier_categorical_columns:
        classifier_frame[col] = classifier_frame[col].fillna("Special: Missings")
    check_shutdown()
    probabilities = np.asarray(bundle.classifier.predict_proba(classifier_frame), dtype=np.float64)
    check_shutdown()
    processed = bundle.preprocessor.transform(prepared, check_shutdown)
    predictions = {target: [] for target in bundle.model.target_names}
    bundle.model.eval()
    with torch.no_grad():
        for start in range(0, len(processed), bundle.spec.infer_batch_size):
            check_shutdown()
            values = processed.iloc[start : start + bundle.spec.infer_batch_size].to_numpy(
                dtype=np.float32
            )
            tensor = torch.tensor(values, dtype=torch.float32).to(bundle.device, non_blocking=True)
            outputs = bundle.model.predict(tensor)
            for target, output in outputs.items():
                predictions[target].append(output.detach().cpu().numpy())
    check_shutdown()
    regression = {
        target: np.concatenate(parts).astype(np.float64, copy=False)
        for target, parts in predictions.items()
    }
    return combine_scores(result, probabilities, regression, bundle.calibrators)

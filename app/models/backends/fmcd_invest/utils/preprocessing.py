"""
Преобразования признаков INVEST по сохранённым параметрам ноутбука.
"""

import numpy as np
import pandas as pd


class FeaturePreprocessor:
    def __init__(self, config, expected_columns):
        self.config = config
        self.params = config.get("params")
        if not isinstance(self.params, dict):
            raise ValueError("Feature preprocessor: params должен быть словарём")
        self.num_cols = list(config["num_cols"])
        self.cat_cols = list(config["cat_cols"])
        self.expected_columns = list(expected_columns)

    def transform(self, raw_frame, check_shutdown):
        frame = raw_frame.loc[:, self.num_cols + self.cat_cols].copy()
        for step in (
            "fill_missing",
            "winsorize_features",
            "transform_features",
            "scale_features",
            "encode_categorical",
        ):
            check_shutdown()
            if self.config[f"step_{step}"]:
                frame = getattr(self, step)(frame)
        missing = set(self.expected_columns) - set(frame.columns)
        extra = set(frame.columns) - set(self.expected_columns)
        if missing or extra:
            raise ValueError(
                f"Колонки после preprocessing: missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}"
            )
        frame = frame.loc[:, self.expected_columns]
        if len(frame.select_dtypes(exclude=[np.number]).columns):
            raise TypeError("После preprocessing остались нечисловые признаки")
        if frame.isna().any().any():
            raise ValueError("После preprocessing остались пропуски")
        return frame.astype(np.float32)

    def fill_missing(self, frame):
        flags = {}
        for col in self.num_cols:
            if self.config.get("add_missing_flags", True):
                flags[f"{col}_missing"] = frame[col].isna().astype("int8")
            frame[col] = frame[col].fillna(self.params["medians"][col])
        if flags:
            frame = pd.concat([frame, pd.DataFrame(flags, index=frame.index)], axis=1)
        missing = self.config.get("missing_cat_value", "Special missing value")
        for col in self.cat_cols:
            if isinstance(frame[col].dtype, pd.CategoricalDtype):
                if missing not in frame[col].cat.categories:
                    frame[col] = frame[col].cat.add_categories([missing])
            frame[col] = frame[col].fillna(missing)
        return frame

    def winsorize_features(self, frame):
        for col in self.num_cols:
            params = self.params["outlier_params"][col]
            lower = params["lower"] if params["lower"] is not None else -np.inf
            upper = params["upper"] if params["upper"] is not None else np.inf
            frame[col] = np.clip(frame[col].to_numpy(dtype=float), lower, upper)
        return frame

    def transform_features(self, frame):
        for col in self.num_cols:
            params = self.params["transform_params"][col]
            method = params["method"]
            values = frame[col].to_numpy(dtype=float)
            if method == "signed_log":
                frame[col] = np.sign(values) * np.log1p(np.abs(values))
            elif method == "asinh":
                frame[col] = np.arcsinh(values / params["scale"])
            elif method != "none":
                raise ValueError(f"Неизвестный feature transform: {method}, {col}")
        return frame

    def scale_features(self, frame):
        for col in self.num_cols:
            params = self.params["scaler_params"][col]
            method = params["method"]
            values = frame[col].to_numpy(dtype=float)
            if method == "robust":
                frame[col] = (values - params["center"]) / params["scale"]
            elif method == "minmax":
                frame[col] = np.clip((values - params["min"]) / params["scale"], 0.0, 1.0)
            elif method != "none":
                raise ValueError(f"Неизвестный feature scaler: {method}, {col}")
        return frame

    def encode_categorical(self, frame):
        method = self.config.get("cat_encode_method", "label")
        for col in self.cat_cols:
            values = frame[col].astype(str).str.strip()
            params = self.params["enc_params"][col]
            if method == "label":
                frame[col] = values.map(params["mapping"]).fillna(0).astype("int64")
            elif method == "frequency":
                frame[col] = values.map(params["freq_map"]).fillna(0.0).astype("float32")
            elif method == "one_hot":
                dummies = pd.get_dummies(values, prefix=col, dtype="int8")
                dummies = dummies.reindex(columns=params["ohe_columns"], fill_value=np.int8(0))
                frame = pd.concat([frame.drop(columns=[col]), dummies], axis=1)
            else:
                raise ValueError(f"Неизвестный categorical encoding: {method}")
        return frame


def prepare_data_for_score(frame, numerical_columns, categorical_columns, fill_columns):
    """
    Повторяет zero-fill перед cast; вход содержит исходные признаки из Parquet.
    """
    result = frame.copy()
    for col in fill_columns:
        # Spark fillna с числом пропускает строковые и логические колонки.
        if pd.api.types.is_numeric_dtype(result[col]) and not pd.api.types.is_bool_dtype(
            result[col]
        ):
            result[col] = result[col].fillna(0)
    for col in numerical_columns:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("float64")
    for col in categorical_columns:
        result[col] = result[col].map(
            lambda value: (
                None
                if pd.isna(value)
                else str(value).lower()
                if isinstance(value, (bool, np.bool_))
                else str(value)
            )
        )
    return result

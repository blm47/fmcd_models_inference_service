"""
Pipeline INVEST: CatBoost, калибровка, нейросетевой stacking и итоговые скоры.
"""

import gc
import sys

from app.models.contracts import ModelBundle, column_names


class FMCDInvestRunner(ModelBundle):
    def __init__(self, spec, logger):
        super().__init__(spec, logger)
        self.classifier = None
        self.model = None
        self.preprocessor = None
        self.calibrators = None
        self.device = None
        self._closed = False
        self._loaded = False

    def load(self) -> None:
        if self._closed or self.classifier is not None:
            raise RuntimeError("Для новой загрузки требуется новый экземпляр pipeline")
        if self.spec.options:
            raise ValueError("Неизвестные options для fmcd_invest")
        root = self.spec.artifacts_dir
        classifier_path = root / "catboost_multilabel_model_v1_optimized.cbm"
        if not classifier_path.is_file():
            raise FileNotFoundError(f"Не найден артефакт INVEST: {classifier_path}")

        import numpy as np
        import pandas as pd
        import torch
        from catboost import CatBoostClassifier

        from .utils.artifacts import load_json, load_stacking_model
        from .utils.preprocessing import FeaturePreprocessor

        self.device = torch.device(self.spec.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA задана в YAML, но недоступна на поде")
        self.logger.info(f"Загрузка pipeline FMCD INVEST из {root}")
        self.classifier = CatBoostClassifier()
        self.classifier.load_model(str(classifier_path))
        self.classifier_columns = list(
            column_names(self.classifier.feature_names_, "CatBoost features")
        )
        reference = pd.read_excel(root / "fmcd_reference.xlsx", index_col=0)
        reference = reference.loc[reference.is_applicable_for_modeling == "YES"]
        if not reference.index.is_unique:
            raise ValueError("В fmcd_reference.xlsx повторяются имена признаков")
        report = pd.read_excel(root / "mrmr_report_v2.xlsx", index_col=0)
        selected = report.loc[report.in_final_selection.eq(True), "feature"].tolist()
        column_names(selected, "MRMR features")
        self.required_columns = tuple(dict.fromkeys(self.classifier_columns + selected))
        if set(self.required_columns) & set(self.spec.id_cols):
            raise ValueError("Ключи не должны входить в признаки модели")
        numeric = set(reference.index[reference.modeling_type == "numerical"])
        categorical = set(reference.index[reference.modeling_type == "categorical"])
        untyped = set(self.required_columns) - numeric - categorical
        if untyped:
            raise ValueError(f"Нет modeling_type для признаков: {sorted(untyped)[:10]}")
        self.numeric_columns = [col for col in self.required_columns if col in numeric]
        self.categorical_columns = [col for col in self.required_columns if col in categorical]
        prefixes = (
            "Частота",
            "Сальдо",
            "Диверсификация",
            "Факт",
            "Регулярность",
            "Средн",
            "Оборот",
        )
        fill_mask = reference.description.str.startswith(prefixes, na=False) & reference.tags.isin(
            ["brokerage_deals", "invest_portfolio"]
        )
        self.fill_columns = [
            col for col in reference.index[fill_mask] if col in self.required_columns
        ]
        self.classifier_categorical_columns = [
            col for col in self.classifier_columns if col in categorical
        ]
        self.calibrators = load_json(root / "calib_model_weights.json")
        for target in "fmcd":
            for key in ("w", "w0"):
                if not np.isfinite(float(self.calibrators[target][key])):
                    raise ValueError(f"Некорректная калибровка {target}.{key}")
        nn_root = root / "artifacts_fmcd"
        config = load_json(nn_root / "stacking_training_config.json")
        feature_config = load_json(nn_root / "feature_preprocessor.json")
        column_names(config["feature_columns"], "Stacking features")
        raw_columns = list(feature_config["num_cols"]) + list(feature_config["cat_cols"])
        column_names(raw_columns, "Stacking raw features")
        if set(raw_columns) - set(selected):
            raise ValueError("Признаки feature_preprocessor отсутствуют в MRMR selection")
        self.preprocessor = FeaturePreprocessor(feature_config, config["feature_columns"])
        self.model = load_stacking_model(nn_root, config, self.device)
        self.output_columns = tuple(
            [f"{target}_prob_score" for target in "fmcd"]
            + [f"{target}_prob_score_calib" for target in "fmcd"]
            + [f"{target}_pred" for target in "fmcd"]
        )
        self._loaded = True
        self.logger.info(
            f"Pipeline FMCD INVEST загружен: {len(self.required_columns)} признаков, "
            f"device={self.device}"
        )

    def predict_batch(self, frame, check_shutdown):
        if self._closed or not self._loaded:
            raise RuntimeError("Pipeline INVEST не загружен")
        from .utils.inference import predict_invest_batch

        return predict_invest_batch(frame, self, check_shutdown)

    def close(self) -> None:
        if self._closed:
            return
        device = self.device
        self.classifier = None
        self.model = None
        self.preprocessor = None
        self.calibrators = None
        self.device = None
        self._loaded = False
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and device is not None and device.type == "cuda":
            if torch.cuda.is_initialized():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        self._closed = True

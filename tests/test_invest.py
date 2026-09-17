"""
Проверки перенесённого INVEST на синтетических признаках и артефактах.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from app.models.backends.fmcd_invest.runner import FMCDInvestRunner
from app.models.backends.fmcd_invest.utils.inference import combine_scores
from app.models.backends.fmcd_invest.utils.preprocessing import (
    FeaturePreprocessor,
    prepare_data_for_score,
)
from app.models.contracts import ModelSpec
from tests.logger_stub import make_logger


def feature_config(method="label"):
    return {
        "num_cols": ["n"],
        "cat_cols": ["cat"],
        "step_fill_missing": True,
        "step_winsorize_features": True,
        "step_transform_features": True,
        "step_scale_features": True,
        "step_encode_categorical": True,
        "cat_encode_method": method,
        "params": {
            "medians": {"n": 4.0},
            "outlier_params": {"n": {"lower": -4.0, "upper": 4.0}},
            "transform_params": {"n": {"method": "signed_log"}},
            "scaler_params": {"n": {"method": "robust", "center": 0.0, "scale": 2.0}},
            "enc_params": {
                "cat": {
                    "mapping": {"a": 1, "Special missing value": 2},
                    "freq_map": {"a": 0.3, "Special missing value": 0.2},
                    "ohe_columns": ["cat_a", "cat_Special missing value"],
                }
            },
        },
    }


class InvestPreprocessingTests(unittest.TestCase):
    def test_steps_order_missing_and_unknown_categories(self):
        raw = pd.DataFrame({"n": [None, -100.0, 100.0], "cat": [None, " a ", "unknown"]})
        original = raw.copy(deep=True)
        for method in ("label", "frequency", "one_hot"):
            expected = ["n", "n_missing"] + (
                ["cat_a", "cat_Special missing value"] if method == "one_hot" else ["cat"]
            )
            check = Mock()
            result = FeaturePreprocessor(feature_config(method), expected).transform(raw, check)
            np.testing.assert_allclose(result.n, [np.log(5) / 2, -np.log(5) / 2, np.log(5) / 2])
            self.assertEqual(result.n_missing.tolist(), [1, 0, 0])
            self.assertEqual(list(result.columns), expected)
            self.assertTrue(all(dtype == np.float32 for dtype in result.dtypes))
            if method == "one_hot":
                self.assertEqual(result.cat_a.tolist(), [0, 1, 0])
                self.assertEqual(result["cat_Special missing value"].tolist(), [1, 0, 0])
            else:
                np.testing.assert_allclose(
                    result.cat, [2, 1, 0] if method == "label" else [0.2, 0.3, 0]
                )
            self.assertEqual(check.call_count, 5)
        pd.testing.assert_frame_equal(raw, original)

    def test_zero_fill_precedes_numeric_cast(self):
        raw = pd.DataFrame({"number": [np.nan, 2.0], "text": [None, "bad"], "cat": [None, " A "]})
        result = prepare_data_for_score(raw, ["number", "text"], ["cat"], ["number", "text"])
        self.assertEqual(result.number.tolist(), [0.0, 2.0])
        self.assertTrue(result.text.isna().all())
        self.assertTrue(pd.isna(result.cat.iloc[0]))
        self.assertEqual(result.cat.iloc[1], " A ")

    def test_processed_schema_and_missing_values_are_checked(self):
        raw = pd.DataFrame({"n": [1.0], "cat": ["a"]})
        with self.assertRaisesRegex(ValueError, "missing"):
            FeaturePreprocessor(feature_config(), ["absent"]).transform(raw, lambda: None)
        config = feature_config()
        config["params"]["medians"]["n"] = np.nan
        raw["n"] = np.nan
        with self.assertRaisesRegex(ValueError, "пропуски"):
            FeaturePreprocessor(config, ["n", "cat", "n_missing"]).transform(raw, lambda: None)

    def test_calibration_clipping_and_bankers_rounding(self):
        result = pd.DataFrame({"id": [1, 2, 3]})
        weights = {target: {"w": 1.0, "w0": 0.0} for target in "fmcd"}
        output = combine_scores(
            result,
            np.array([[0.5, 0.5, 0.5], [0, 0, 0], [1, 1, 1]]),
            {
                "F": np.array([5, -1, 7]),
                "M": np.array([-2.25, 10, -3.125]),
                "C": np.array([100, -2, 100]),
                "D": np.array([100, -2, 100]),
            },
            weights,
        )
        self.assertEqual(output.f_pred.tolist(), [2, 0, 7])
        self.assertEqual(output.c_pred.tolist(), [31, 0, 62])
        self.assertEqual(output.d_pred.tolist(), [6, 0, 13])
        self.assertEqual(output.m_pred.tolist(), [-1.12, 0, -3.12])
        np.testing.assert_array_equal(output.f_prob_score, output.c_prob_score)


HAS_MODELS = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "catboost", "openpyxl")
)


@unittest.skipUnless(HAS_MODELS, "Нужны torch, catboost и openpyxl")
class InvestArtifactTests(unittest.TestCase):
    def setUp(self):
        import torch
        from catboost import CatBoostClassifier

        from app.models.backends.fmcd_invest.utils.networks import (
            FMCDSingleTargetNet,
            FMCDSingleTargetStackingModel,
        )

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = ModelSpec(
            "invest", "fmcd_invest", self.root, ("counterparty_id", "report_dt"), "cpu", 2, 5
        )
        train = pd.DataFrame(
            {"n": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "cat": ["a", "b", "a", "b", "a", "b"]}
        )
        classifier = CatBoostClassifier(
            iterations=2,
            depth=2,
            loss_function="MultiLogloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        )
        classifier.fit(
            train,
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]),
            cat_features=["cat"],
        )
        classifier.save_model(str(self.root / "catboost_multilabel_model_v1_optimized.cbm"))
        pd.DataFrame(
            {
                "is_applicable_for_modeling": ["YES", "YES"],
                "modeling_type": ["numerical", "categorical"],
                "description": ["Частота сделок", "Категория"],
                "tags": ["brokerage_deals", "other"],
            },
            index=["n", "cat"],
        ).to_excel(self.root / "fmcd_reference.xlsx")
        pd.DataFrame({"in_final_selection": [True, True], "feature": ["n", "cat"]}).to_excel(
            self.root / "mrmr_report_v2.xlsx"
        )
        (self.root / "calib_model_weights.json").write_text(
            json.dumps({target: {"w": 1.0, "w0": 0.0} for target in "fmcd"}), encoding="utf-8"
        )
        nn_root = self.root / "artifacts_fmcd"
        nn_root.mkdir()
        (nn_root / "feature_preprocessor.json").write_text(
            json.dumps(feature_config()), encoding="utf-8"
        )
        (nn_root / "stacking_training_config.json").write_text(
            json.dumps({"feature_columns": ["cat", "n_missing", "n"], "meta_hidden_dims": [4, 3]}),
            encoding="utf-8",
        )
        bases = {}
        for target in "FMCD":
            params = {
                "input_dim": 3,
                "target_name": target,
                "backbone_dims": [4],
                "head_dim": 2,
                "dropout": 0.0,
            }
            folder = nn_root / "experiments_single_target" / target / "artifacts"
            folder.mkdir(parents=True)
            (folder / "fmcd_nn_artifacts.json").write_text(
                json.dumps({"model_init_params": params}), encoding="utf-8"
            )
            bases[target] = FMCDSingleTargetNet(**params)
        torch.manual_seed(7)
        self.source_model = FMCDSingleTargetStackingModel(bases, meta_hidden_dims=(4, 3)).eval()
        self.state_path = (
            nn_root / "experiments_stacking" / "artifacts" / "fmcd_stacking_best_state.pt"
        )
        self.state_path.parent.mkdir(parents=True)
        torch.save(self.source_model.state_dict(), self.state_path)
        self.runner = FMCDInvestRunner(self.spec, make_logger())
        self.addCleanup(self.runner.close)

    def test_load_predict_chunking_and_lifecycle(self):
        import torch

        from app.models.inference import run_inference_on_chunk

        self.runner.load()
        for key, value in self.source_model.state_dict().items():
            torch.testing.assert_close(value, self.runner.model.state_dict()[key])
        frame = pd.DataFrame(
            {
                "counterparty_id": [8, 3, 6],
                "report_dt": ["2025-12-31"] * 3,
                "n": [None, 3.0, -1.0],
                "cat": [None, "a", "unknown"],
            },
            index=[9, 2, 2],
        )
        original = frame.copy(deep=True)
        check = Mock()
        result = run_inference_on_chunk(frame, self.runner, 2, check)
        self.assertEqual(len(result), 3)
        self.assertEqual(
            list(result.columns), list(self.spec.id_cols) + list(self.runner.output_columns)
        )
        self.assertEqual(result.counterparty_id.tolist(), [8, 3, 6])
        self.assertFalse(result.isna().any().any())
        pd.testing.assert_frame_equal(frame, original)
        whole = self.runner.predict_batch(frame.reset_index(drop=True), check)
        np.testing.assert_allclose(result.iloc[:, 2:], whole.iloc[:, 2:], atol=1e-6)
        empty = self.runner.predict_batch(frame.iloc[:0], check)
        self.assertEqual(list(empty.columns), list(result.columns))
        self.runner.close()
        self.runner.close()
        with self.assertRaises(RuntimeError):
            self.runner.predict_batch(frame, check)

    def test_checkpoint_without_base_weights_is_rejected(self):
        import torch

        torch.save(
            {
                key: value
                for key, value in self.source_model.state_dict().items()
                if not key.startswith("base_models.F.")
            },
            self.state_path,
        )
        with self.assertRaisesRegex(ValueError, "base model F"):
            self.runner.load()

    def test_inverse_transform_after_normalization(self):
        import torch

        from app.models.backends.fmcd_invest.utils.networks import FMCDSingleTargetNet

        model = FMCDSingleTargetNet(
            1,
            "M",
            backbone_dims=(2,),
            target_transform_params={"M": {"method": "signed_log"}},
            target_normalization_params={
                "M": {"method": "standard", "per_column": {"M": {"center": 1.0, "scale": 2.0}}}
            },
        )
        values = torch.tensor([-1.0, 0.0, 1.0])
        torch.testing.assert_close(
            model.inverse_target(values),
            torch.tensor([-np.expm1(1), np.expm1(1), np.expm1(3)], dtype=torch.float32),
        )

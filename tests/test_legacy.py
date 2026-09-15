"""
Регрессия формул и освобождения ресурсов legacy с заглушкой PyTorch.
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from logger_stub import make_logger

from app.models.backends.fmcd_cc_dc.runner import FMCDCardsRunner
from app.models.contracts import ModelSpec


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.values


class LegacyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.spec = ModelSpec("cc", "fmcd_cc_dc", directory.name, ("id",), "cpu", 2, 3)
        self.bundle = FMCDCardsRunner(self.spec, make_logger())
        self.schema = types.SimpleNamespace(
            d_cols=["d"],
            f_cols=["f"],
            m_cols=["m"],
            c_cols=["c"],
            num_mcg=1,
            num_cols=["num"],
            cat_cols=["cat"],
            num_numerical=1,
            num_categorical=1,
        )
        self.model = Mock()
        self.torch = types.SimpleNamespace(
            device=lambda value: types.SimpleNamespace(type=value.split(":")[0]),
            serialization=types.SimpleNamespace(add_safe_globals=Mock()),
            load=Mock(return_value=self.model),
            from_numpy=lambda x: x,
            cuda=Mock(),
            no_grad=nullcontext,
            sigmoid=lambda x: Tensor(1 / (1 + np.exp(-x.values))),
        )
        self.production = types.SimpleNamespace(
            ProdSchema=types.SimpleNamespace(from_json_local=Mock(return_value=self.schema)),
            load_freq_counts_local=Mock(return_value=[np.array([1, 2])]),
        )
        self.modules = {
            "torch": self.torch,
            "app.models.backends.fmcd_cc_dc.utils.fmcd.data.production": self.production,
            "app.models.backends.fmcd_cc_dc.utils.fmcd.model.fmcd_model": types.SimpleNamespace(
                FMCDModel=type("FMCDModel", (), {})
            ),
        }
        (self.spec.artifacts_dir / "calibrators.json").write_text(
            json.dumps({"d": {"intercept": 0.25, "coef": 1.5}}), encoding="utf-8"
        )

    def test_load_contract_and_idempotent_close(self):
        with patch.dict(sys.modules, self.modules):
            self.bundle.load()
            self.assertEqual(self.bundle.required_columns, ("num", "cat"))
            self.assertEqual(
                self.bundle.output_columns,
                ("d_prob_d", "f_pred_f", "m_pred_m", "c_pred_c", "d_count_pred"),
            )
            self.bundle.close()
            self.bundle.close()
        self.assertIsNone(self.bundle.model)
        self.assertIsNone(self.bundle.schema)
        self.assertIsNone(self.bundle.calibrators)
        self.torch.cuda.empty_cache.assert_not_called()

    def test_partial_load_releases_model_after_preprocessing_failure(self):
        self.production.load_freq_counts_local.side_effect = RuntimeError("encoding failed")
        with patch.dict(sys.modules, self.modules):
            with self.assertRaisesRegex(RuntimeError, "encoding failed"):
                self.bundle.load()
            self.bundle.close()
        self.assertIsNone(self.bundle.model)
        self.assertIsNone(self.bundle.schema)

    def test_cuda_unavailable_has_no_cpu_fallback(self):
        self.bundle.spec = ModelSpec("cc", "fmcd_cc_dc", ".", ("id",), "cuda:0", 2, 3)
        self.torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, self.modules), self.assertRaisesRegex(RuntimeError, "CUDA"):
            self.bundle.load()
        self.torch.load.assert_not_called()

    def test_cuda_cache_cleared_once_at_close(self):
        self.bundle.device = types.SimpleNamespace(type="cuda")
        self.torch.cuda.device.return_value = nullcontext()
        self.torch.cuda.is_initialized.return_value = True
        with patch.dict(sys.modules, self.modules):
            self.bundle.close()
            self.bundle.close()
        self.torch.cuda.empty_cache.assert_called_once_with()

    def test_preprocessing_nan_and_unknown_categories_preserves_input(self):
        self.schema.cat_cardinalities = [3]
        self.torch.zeros = lambda *shape: np.zeros(shape)
        module_spec = importlib.util.spec_from_file_location(
            "legacy_preprocessing_test", "app/models/backends/fmcd_cc_dc/utils/pandas_to_fmcd.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        modules = {
            **self.modules,
            "app.models.backends.fmcd_cc_dc.utils.fmcd.data.base": types.SimpleNamespace(
                FMCDBatch=types.SimpleNamespace
            ),
        }
        with patch.dict(sys.modules, modules):
            module_spec.loader.exec_module(module)
        frame = pd.DataFrame({"num": [np.nan, 1.5, -2, 0], "cat": [np.nan, -1, 99, 2]})
        original = frame.copy(deep=True)
        batch = module.pandas_chunk_to_fmcd_batch(frame, self.schema, ["num"], ["cat"])
        np.testing.assert_array_equal(batch.num_features[:, 0], [0, 1.5, -2, 0])
        np.testing.assert_array_equal(batch.missing_mask[:, 0], [0, 1, 1, 1])
        np.testing.assert_array_equal(batch.cat_features[:, 0], [3, 0, 3, 2])
        pd.testing.assert_frame_equal(frame, original)

    def test_formula_outputs_for_single_row_and_extreme_logits(self):
        spec = importlib.util.spec_from_file_location(
            "legacy_formula_test", "app/models/backends/fmcd_cc_dc/utils/inference.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, self.modules):
            spec.loader.exec_module(module)
            self.bundle.load()
        for logits in ([0.0], [-100.0, 0.0, 100.0]):
            frame = pd.DataFrame({"id": list(range(len(logits)))}, index=[7] * len(logits))
            outputs = types.SimpleNamespace(
                d_multilabel_logits=Tensor(np.array(logits).reshape(-1, 1)),
                f_value_pred=Tensor([[1.0]] * len(logits)),
                m_value_pred=Tensor([[2.0]] * len(logits)),
                c_value_pred=Tensor([[3.0]] * len(logits)),
                d_count_pred=Tensor([[0.5]] * len(logits)),
            )
            self.model.return_value = outputs
            converter = types.SimpleNamespace(pandas_chunk_to_fmcd_batch=Mock(return_value=Mock()))
            with (
                patch.dict(
                    sys.modules, {"app.models.backends.fmcd_cc_dc.utils.pandas_to_fmcd": converter}
                ),
                np.errstate(over="ignore"),
            ):
                result = module.predict_fmcd_batch(frame, self.bundle, lambda: None)
                probability = np.clip(
                    1 / (1 + np.exp(-np.array(logits, dtype=np.float32))), 1e-7, 1 - 1e-7
                )
                expected = 1 / (1 + np.exp(-(0.25 + 1.5 * np.log(probability / (1 - probability)))))
            np.testing.assert_allclose(result.d_prob_d, expected, rtol=1e-6)
            for col, value in (("f_pred_f", 1), ("m_pred_m", 2), ("c_pred_c", 3)):
                np.testing.assert_allclose(
                    result[col], expected * np.expm1(np.float32(value)), rtol=1e-6
                )
            np.testing.assert_allclose(result.d_count_pred, np.expm1(np.float32(0.5)), rtol=1e-6)
            self.assertEqual(result.id.tolist(), frame.id.tolist())

"""
Проверки лёгких контрактов MODEL-001 без библиотек моделей и настоящих весов.
"""

import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.models.contracts import ModelBundle, ModelSpec


class ModelContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.spec = ModelSpec(
            name="fmcd_credit_cards",
            backend="fmcd_cc_dc",
            artifacts_dir=Path(directory.name),
            id_cols=["customer_id", "report_dt"],
            device="cpu",
            infer_batch_size=10,
            parquet_read_chunk_size=3,
        )

    def test_contract_imports_do_not_import_heavy_libraries(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from app.models.contracts import ModelSpec; "
                "from app.models.registry import BACKENDS, create_bundle; "
                "[create_bundle(ModelSpec('test', backend, 'missing', ('id',), 'cpu', 1, 1), "
                "object()) for backend in BACKENDS]; "
                "assert not ({'torch', 'fmcd', 'catboost', 'pandas'} & sys.modules.keys())",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_spec_does_not_read_artifacts_or_require_cuda(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("чтение артефактов")):
            spec = replace(self.spec, artifacts_dir="missing/model", device="cuda:0")
        self.assertEqual(spec.id_cols, ("customer_id", "report_dt"))
        self.assertEqual(spec.artifacts_dir, Path("missing/model"))

    def test_invalid_spec_is_rejected(self):
        for field, value in (
            ("name", " "),
            ("backend", None),
            ("device", ""),
            ("artifacts_dir", ""),
            ("id_cols", []),
            ("id_cols", ["id", "id"]),
            ("infer_batch_size", True),
            ("infer_batch_size", 0),
            ("parquet_read_chunk_size", -1),
            ("options", []),
            ("options", {"device": "cpu"}),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(self.spec, **{field: value})

    def test_bundle_requires_all_lifecycle_methods(self):
        with self.assertRaises(TypeError):
            ModelBundle(self.spec, object())

        class IncompleteBundle(ModelBundle):
            def load(self):
                pass

            def predict_batch(self, frame, check_shutdown):
                return frame

        with self.assertRaises(TypeError):
            IncompleteBundle(self.spec, object())

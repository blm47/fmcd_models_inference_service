"""
Адаптер FMCD: один экземпляр и комплект ресурсов на одну задачу.
"""

import gc
import json
import sys
from pathlib import Path, PureWindowsPath

from app.models.contracts import ModelBundle


class FMCDCardsRunner(ModelBundle):
    def __init__(self, spec, logger):
        super().__init__(spec, logger)
        self.model = None
        self.schema = None
        self.calibrators = None
        self.device = None
        self._closed = False
        self.id_cols = spec.id_cols

    def _path(self, option: str, default: str) -> str:
        value = self.spec.options.get(option, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{option}: ожидается относительный путь")
        relative = Path(value)
        windows = PureWindowsPath(value)
        directory = self.spec.artifacts_dir.resolve()
        path = (directory / relative).resolve()
        if windows.drive or windows.root or not path.is_relative_to(directory):
            raise ValueError(f"{option}: путь должен быть внутри artifacts_dir")
        return str(path)

    def load(self) -> None:
        if self._closed or self.model is not None:
            raise RuntimeError("Для новой загрузки требуется новый экземпляр pipeline")
        allowed_options = {"weights_file", "schema_file", "calibrators_file", "freq_encoding_file"}
        if set(self.spec.options) - allowed_options:
            raise ValueError("Неизвестные options для fmcd_cc_dc")

        import torch

        from .utils.fmcd.data.production import ProdSchema, load_freq_counts_local
        from .utils.fmcd.model.fmcd_model import FMCDModel

        self.device = torch.device(self.spec.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA задана в YAML, но недоступна на поде")
        self.schema = ProdSchema.from_json_local(self._path("schema_file", "schema.json"))
        schema = self.schema
        for prefix in ("d", "f", "m", "c"):
            cols = getattr(schema, f"{prefix}_cols") or [
                f"{prefix}_{k}" for k in range(schema.num_mcg)
            ]
            if len(cols) != schema.num_mcg:
                raise ValueError(f"Некорректная длина {prefix}_cols в schema")
            setattr(self, f"{prefix}_cols", cols)
        self.num_cols = schema.num_cols or [f"num_{i}" for i in range(schema.num_numerical)]
        self.cat_cols = schema.cat_cols or [f"cat_{i}" for i in range(schema.num_categorical)]
        if (
            len(self.num_cols) != schema.num_numerical
            or len(self.cat_cols) != schema.num_categorical
        ):
            raise ValueError("Количество признаков не соответствует schema")
        self.required_columns = tuple(dict.fromkeys(self.num_cols + self.cat_cols))
        self.output_columns = tuple(
            [f"d_prob_{col}" for col in self.d_cols]
            + [f"f_pred_{col}" for col in self.f_cols]
            + [f"m_pred_{col}" for col in self.m_cols]
            + [f"c_pred_{col}" for col in self.c_cols]
            + ["d_count_pred"]
        )
        torch.serialization.add_safe_globals([FMCDModel])
        self.model = torch.load(
            self._path("weights_file", "model.pth"), map_location=self.device, weights_only=False
        )
        self.model.to(self.device)
        self.model.eval()
        freq_counts = load_freq_counts_local(
            self._path("freq_encoding_file", "freq_encoding.parquet"), schema
        )
        if len(freq_counts) != schema.num_categorical:
            raise ValueError("Количество frequency encoding не соответствует schema")
        for i, counts in enumerate(freq_counts):
            self.model.cat_processor.set_frequency_encoding(i, torch.from_numpy(counts))
        with open(self._path("calibrators_file", "calibrators.json"), encoding="utf-8") as source:
            self.calibrators = json.load(source)
        for col in self.d_cols:
            if not {"intercept", "coef"} <= self.calibrators[col].keys():
                raise ValueError(f"Неполный calibrator для {col}")

    def predict_batch(self, frame, check_shutdown):
        if self._closed or self.model is None or self.calibrators is None:
            raise RuntimeError("Pipeline не загружен")
        from app.models.backends.fmcd_cc_dc.utils.inference import predict_fmcd_batch

        return predict_fmcd_batch(frame, self, check_shutdown)

    def close(self) -> None:
        if self._closed:
            return
        device = self.device
        self.model = None
        self.schema = None
        self.calibrators = None
        self.device = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and device is not None and device.type == "cuda":
            if torch.cuda.is_initialized():
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        self._closed = True

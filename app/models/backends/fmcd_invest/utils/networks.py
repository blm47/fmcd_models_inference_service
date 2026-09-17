"""
Архитектуры INVEST с именами слоёв, совместимыми со stacking state_dict.
"""

import math
from copy import deepcopy

import numpy as np
import torch
from torch import nn


class FMCDSingleTargetNet(nn.Module):
    def __init__(
        self,
        input_dim,
        target_name,
        backbone_dims=(512, 256, 128),
        head_dim=32,
        dropout=0.2,
        c_max_val=62.0,
        d_max_val=13.0,
        target_transform_params=None,
        target_normalization_params=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.target_name = str(target_name).strip().upper()
        if self.target_name not in ("F", "M", "C", "D"):
            raise ValueError("target_name должен быть F, M, C или D")
        self.backbone_dims = tuple(int(dim) for dim in backbone_dims)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)
        self.c_max_val = float(c_max_val)
        self.d_max_val = float(d_max_val)
        self.target_transform_params = deepcopy(target_transform_params or {})
        self.target_normalization_params = deepcopy(target_normalization_params or {})
        layers = []
        in_dim = self.input_dim
        for out_dim in self.backbone_dims:
            layers.extend([nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU()])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_dim = out_dim
        self.backbone = nn.Sequential(*layers)
        head = [
            nn.Linear(self.backbone_dims[-1], self.head_dim),
            nn.GELU(),
            nn.Linear(self.head_dim, 1),
        ]
        if self._uses_sigmoid_head():
            head.append(nn.Sigmoid())
        self.head = nn.Sequential(*head)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _target_transform_method(self):
        return (
            str(self.target_transform_params.get(self.target_name, {}).get("method", "none"))
            .strip()
            .lower()
        )

    def _target_is_normalized(self):
        return str(
            self.target_normalization_params.get(self.target_name, {}).get("method", "none")
        ).strip().lower() in ("robust", "standard")

    def _uses_sigmoid_head(self):
        return (
            not self._target_is_normalized()
            and self._target_transform_method() == "log1p_normalized"
        )

    def _inverse_target_normalization(self, y):
        params = self.target_normalization_params.get(self.target_name, {})
        method = str(params.get("method", "none")).strip().lower()
        if method == "none":
            return y
        if method not in ("robust", "standard"):
            raise ValueError(f"Неизвестный target normalization: {method}")
        column = params.get("per_column", {}).get(self.target_name)
        if column is None:
            raise ValueError(f"Нет normalization params для {self.target_name}")
        return y * y.new_tensor(float(column["scale"])) + y.new_tensor(float(column["center"]))

    def inverse_target(self, y):
        y = self._inverse_target_normalization(y)
        params = self.target_transform_params.get(self.target_name, {"method": "none"})
        method = str(params.get("method", "none")).strip().lower()
        if method in ("none", "nb", "zinb"):
            return y
        if method == "log1p":
            return torch.clamp(torch.expm1(y), min=0.0)
        if method == "log1p_normalized":
            log1p_max = params.get("log1p_max")
            if log1p_max is None:
                max_val = params.get("max_val")
                if max_val is None:
                    max_val = self.c_max_val if self.target_name == "C" else self.d_max_val
                log1p_max = math.log1p(float(max_val))
            return torch.clamp(torch.expm1(y * y.new_tensor(float(log1p_max))), min=0.0)
        if method == "sqrt":
            return torch.clamp(y**2, min=0.0)
        if method == "signed_log":
            return torch.sign(y) * torch.expm1(torch.abs(y))
        if method in ("asinh", "asinh_quantile"):
            scale = float(params.get("scale", 1.0))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            return y.new_tensor(scale) * torch.sinh(y)
        raise ValueError(f"Неизвестный target transform: {method}")

    def forward_features(self, x):
        return self.backbone(x)

    def forward_head(self, backbone_features, return_embedding=False):
        embedding = self.head[1](self.head[0](backbone_features))
        output = self.head[2](embedding)
        if len(self.head) > 3:
            output = self.head[3](output)
        output = output.squeeze(1)
        return (output, embedding) if return_embedding else output

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def forward_with_embedding(self, x):
        return self.forward_head(self.forward_features(x), return_embedding=True)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return self.inverse_target(self(x))


class FMCDSingleTargetStackingModel(nn.Module):
    def __init__(
        self,
        base_models,
        target_names=("F", "M", "C", "D"),
        meta_hidden_dims=(64, 32),
        meta_dropout_first=0.20,
        meta_dropout_second=0.10,
        freeze_base_models=True,
    ):
        super().__init__()
        self.target_names = tuple(str(name).strip().upper() for name in target_names)
        if self.target_names != ("F", "M", "C", "D") or len(meta_hidden_dims) != 2:
            raise ValueError("Stacking ожидает targets F/M/C/D и две meta hidden dimensions")
        self.base_models = nn.ModuleDict({name: base_models[name] for name in self.target_names})
        self.meta_hidden_dims = tuple(int(dim) for dim in meta_hidden_dims)
        self.meta_dropout_first = float(meta_dropout_first)
        self.meta_dropout_second = float(meta_dropout_second)
        self.freeze_base_models_flag = bool(freeze_base_models)
        self.embedding_dims = {name: int(base_models[name].head_dim) for name in self.target_names}
        self.meta_input_dim = sum(self.embedding_dims.values()) + len(self.target_names)
        dim1, dim2 = self.meta_hidden_dims
        self.meta_backbone = nn.Sequential(
            nn.BatchNorm1d(self.meta_input_dim),
            nn.Linear(self.meta_input_dim, dim1),
            nn.BatchNorm1d(dim1),
            nn.GELU(),
            nn.Dropout(self.meta_dropout_first),
            nn.Linear(dim1, dim2),
            nn.GELU(),
            nn.Dropout(self.meta_dropout_second),
        )
        self.output_heads = nn.ModuleDict({name: nn.Linear(dim2, 1) for name in self.target_names})
        self.output_activations = nn.ModuleDict(
            {
                name: nn.Sigmoid() if base_models[name]._uses_sigmoid_head() else nn.Identity()
                for name in self.target_names
            }
        )
        self._init_meta_weights()
        if self.freeze_base_models_flag:
            self.freeze_base_models()

    def _init_meta_weights(self):
        for module in self.meta_backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.output_heads.values():
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            nn.init.zeros_(module.bias)

    def freeze_base_models(self):
        self.freeze_base_models_flag = True
        for model in self.base_models.values():
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad = False

    def unfreeze_base_models(self):
        self.freeze_base_models_flag = False
        for model in self.base_models.values():
            for parameter in model.parameters():
                parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base_models_flag:
            for model in self.base_models.values():
                model.eval()
        return self

    def _collect_base_outputs(self, x):
        embeddings, predictions = [], []
        for name in self.target_names:
            model = self.base_models[name]
            if self.freeze_base_models_flag:
                with torch.no_grad():
                    pred, emb = model.forward_with_embedding(x)
            else:
                pred, emb = model.forward_with_embedding(x)
            embeddings.append(emb)
            predictions.append(pred.unsqueeze(1))
        return embeddings, predictions

    def build_meta_input(self, x):
        embeddings, predictions = self._collect_base_outputs(x)
        return torch.cat([*embeddings, *predictions], dim=1)

    def forward(self, x):
        hidden = self.meta_backbone(self.build_meta_input(x))
        return tuple(
            self.output_activations[name](self.output_heads[name](hidden)).squeeze(1)
            for name in self.target_names
        )

    def inverse_F(self, y):
        return self.base_models["F"].inverse_target(y)

    def inverse_M(self, y):
        return self.base_models["M"].inverse_target(y)

    def inverse_C(self, y):
        return self.base_models["C"].inverse_target(y)

    def inverse_D(self, y):
        return self.base_models["D"].inverse_target(y)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        return {
            name: self.base_models[name].inverse_target(output)
            for name, output in zip(self.target_names, self(x), strict=True)
        }

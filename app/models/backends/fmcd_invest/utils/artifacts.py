"""
Загрузка JSON и полного stacking checkpoint INVEST.
"""

import json


def load_json(path):
    with path.open(encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON должен содержать объект: {path}")
    return payload


def load_stacking_model(root, config, device):
    import torch

    from .networks import FMCDSingleTargetNet, FMCDSingleTargetStackingModel

    state_path = root / "experiments_stacking" / "artifacts" / "fmcd_stacking_best_state.pt"
    state = torch.load(state_path, map_location=device, weights_only=True)
    targets = tuple(config.get("target_names", ("F", "M", "C", "D")))
    if targets != ("F", "M", "C", "D"):
        raise ValueError("Stacking target_names должны быть F/M/C/D в этом порядке")
    base_models = {}
    for target in targets:
        if not any(key.startswith(f"base_models.{target}.") for key in state):
            raise ValueError(f"Stacking checkpoint не содержит веса base model {target}")
        metadata = load_json(
            root / "experiments_single_target" / target / "artifacts" / "fmcd_nn_artifacts.json"
        )
        params = dict(metadata["model_init_params"])
        if int(params["input_dim"]) != len(config["feature_columns"]):
            raise ValueError(f"input_dim модели {target} не совпадает с числом признаков")
        if str(params["target_name"]).strip().upper() != target:
            raise ValueError(f"target_name в metadata не совпадает с каталогом {target}")
        base_models[target] = FMCDSingleTargetNet(**params).to(device)
    model = FMCDSingleTargetStackingModel(
        base_models=base_models,
        target_names=targets,
        meta_hidden_dims=tuple(config.get("meta_hidden_dims", (64, 32))),
        meta_dropout_first=float(config.get("meta_dropout_first", 0.20)),
        meta_dropout_second=float(config.get("meta_dropout_second", 0.10)),
        freeze_base_models=bool(config.get("freeze_base_models", True)),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model

from __future__ import annotations

import torch
import torch.nn as nn

from .zero_init import apply_zero_init, apply_zero_init_transformer_layer

CNN_INITS = ("zero", "kaiming", "xavier", "rezero")
TRANSFORMER_INITS = ("zero", "standard", "kaiming", "xavier", "rezero")


def _reset_norms(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)) or (
            module.__class__.__name__ == "ScalarAffine"
        ):
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


@torch.no_grad()
def reset_rezero_alphas(model: nn.Module) -> None:
    """Ensure every residual gate α is exactly 0 (identity at initialization)."""
    for module in model.modules():
        alpha = getattr(module, "alpha", None)
        if isinstance(alpha, nn.Parameter):
            alpha.zero_()
        resweight = getattr(module, "resweight", None)
        if isinstance(resweight, nn.Parameter):
            resweight.zero_()


@torch.no_grad()
def init_cnn(model: nn.Module, name: str) -> nn.Module:
    if name == "zero":
        return apply_zero_init(model)

    # ReZero keeps Kaiming/He weights; dynamical isometry comes from α = 0, not from
    # zeroing the last conv (ZerO / Fixup). BN stays (unlike FixUp).
    weight_init = "kaiming" if name == "rezero" else name
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if weight_init == "kaiming":
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif weight_init == "xavier":
                nn.init.xavier_normal_(module.weight)
            else:
                raise ValueError(f"Unknown init={name!r}; choose from {CNN_INITS}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    _reset_norms(model)
    if name == "rezero":
        if not any(getattr(m, "alpha", None) is not None for m in model.modules()):
            raise ValueError(
                "init='rezero' requires ResNet(..., rezero=True); "
                "build_model should pass rezero=(init == 'rezero')"
            )
        reset_rezero_alphas(model)
    return model


@torch.no_grad()
def init_transformer_layers(layers: nn.ModuleList, name: str) -> None:
    """
    Initialise attention + feed-forward of every encoder layer.
    'standard' keeps PyTorch defaults; 'rezero' only resets α (weights already set).
    """
    if name in ("standard", "rezero"):
        if name == "rezero":
            reset_rezero_alphas(layers)
        return
    for layer in layers:
        if name == "zero":
            apply_zero_init_transformer_layer(layer)
            continue

        matrices = [
            layer.self_attn.in_proj_weight,
            layer.self_attn.out_proj.weight,
            layer.linear1.weight,
            layer.linear2.weight,
        ]
        for w in matrices:
            if name == "kaiming":
                nn.init.kaiming_normal_(w, nonlinearity="relu")
            elif name == "xavier":
                nn.init.xavier_uniform_(w)
            else:
                raise ValueError(f"Unknown init={name!r}; choose from {TRANSFORMER_INITS}")
        for b in (
            layer.self_attn.in_proj_bias,
            layer.self_attn.out_proj.bias,
            layer.linear1.bias,
            layer.linear2.bias,
        ):
            if b is not None:
                nn.init.zeros_(b)
        _reset_norms(layer)

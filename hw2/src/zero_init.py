"""ZerO initialization (Zhao et al., arXiv:2110.12661), Algorithms 1–2."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from scipy.linalg import hadamard


def partial_identity(out_features: int, in_features: int) -> torch.Tensor:
    """I* from Definition 1: identity clipped or zero-padded to (out, in)."""
    return torch.eye(out_features, in_features)


def zero_matrix(out_features: int, in_features: int) -> torch.Tensor:
    """Algorithm 1 for a dense weight of shape (P, Q) = (out, in)."""
    if out_features <= in_features:
        return partial_identity(out_features, in_features)

    clog_m = math.ceil(math.log2(out_features))
    p = 2 ** (clog_m)
    hadamard_matrix = torch.tensor(hadamard(p)).float() / (2 ** (clog_m / 2))
    init_matrix = partial_identity(out_features, p) @ hadamard_matrix @ partial_identity(p, in_features)
    return init_matrix


def zero_conv_kernel(weight: torch.Tensor) -> None:
    """Algorithm 2: zeros everywhere except the spatial center, filled by Algorithm 1."""
    if weight.ndim != 4:
        raise ValueError(f"Expected (cout, cin, k, k) weight, got {tuple(weight.shape)}")
    cout, cin, kh, kw = weight.shape
    if kh != kw or kh % 2 == 0:
        raise ValueError(f"ZerO expects odd square kernels, got {kh}x{kw}")

    center = kh // 2
    weight.zero_()
    weight[:, :, center, center] = zero_matrix(cout, cin).to(weight)


def _reset_affine(module: nn.Module) -> None:
    if getattr(module, "weight", None) is not None:
        nn.init.ones_(module.weight)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


@torch.no_grad()
def apply_zero_init(model: nn.Module) -> nn.Module:
    """
    ZerO for CNNs: Algorithm 2 for every Conv2d, Algorithm 1 for Linear,
    BN (or its scalar replacement) scale=1 / bias=0, and the last convolution
    of every residual branch set to zero.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if module.groups != 1:
                raise ValueError("ZerO Algorithm 2 assumes groups=1 convolutions")
            zero_conv_kernel(module.weight)
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.copy_(zero_matrix(module.out_features, module.in_features))
            if module.bias is not None:
                module.bias.zero_()
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            _reset_affine(module)
        elif module.__class__.__name__ == "ScalarAffine":
            _reset_affine(module)

    for module in model.modules():
        # print(module)
        last_conv = getattr(module, "last_conv", None)
        if isinstance(last_conv, nn.Conv2d):
            last_conv.weight.zero_()
            if last_conv.bias is not None:
                last_conv.bias.zero_()

    return model


@torch.no_grad()
def apply_zero_init_transformer_layer(layer: nn.TransformerEncoderLayer) -> None:
    """
    ZerO for one Transformer layer: W_Q = I, W_K = W_V = 0, attention output
    projection and feed-forward matrices by Algorithm 1, LayerNorm 1/0.
    """
    attn = layer.self_attn
    if not attn._qkv_same_embed_dim:
        raise ValueError("Expected packed in_proj_weight (kdim == vdim == embed_dim)")
    d = attn.embed_dim

    attn.in_proj_weight.zero_()
    attn.in_proj_weight[:d].copy_(torch.eye(d))
    if attn.in_proj_bias is not None:
        attn.in_proj_bias.zero_()
    attn.out_proj.weight.copy_(zero_matrix(d, d))
    if attn.out_proj.bias is not None:
        attn.out_proj.bias.zero_()

    for linear in (layer.linear1, layer.linear2):
        linear.weight.copy_(zero_matrix(linear.out_features, linear.in_features))
        if linear.bias is not None:
            linear.bias.zero_()

    for norm in (layer.norm1, layer.norm2):
        _reset_affine(norm)


# if __name__ == "__main__":
#     model = resnet18()
#     apply_zero_init(model)
#     print(list(model.state_dict().values()))
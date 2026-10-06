"""Metrics from the ZerO paper: top-1 error, perplexity, stable rank, kernel rank."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from .zero_init import apply_zero_init

RANK_TOL = 1e-4  # tolerance of np.linalg.matrix_rank in the authors' code


def top1_error(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of examples where the argmax prediction is wrong."""
    return 1.0 - (logits.argmax(dim=1) == targets).float().mean().item()


def perplexity(mean_nll: float) -> float:
    return math.exp(min(mean_nll, 50.0))


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}


def as_matrix(weight: torch.Tensor) -> torch.Tensor:
    """Conv kernel (cout, cin, k, k) -> (cout, cin*k*k); matrices unchanged."""
    if weight.ndim == 4:
        return weight.reshape(weight.shape[0], -1)
    if weight.ndim == 2:
        return weight
    raise ValueError(f"Expected 2D or 4D weight, got {tuple(weight.shape)}")


def _svdvals(weight: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(as_matrix(weight.detach()).float())


def stable_rank(weight: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2 = sum(sigma_i^2) / sigma_max^2 (0 for W = 0)."""
    s = _svdvals(weight)
    smax = s.max()
    if smax <= 0:
        return 0.0
    return ((s**2).sum() / smax**2).item()


def matrix_rank(weight: torch.Tensor, tol: float = RANK_TOL) -> int:
    return int((_svdvals(weight) > tol).sum().item())


def rank_stats(weight: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """
    Ranks of the weight and of its residual component W - W_ZerO, where W_ZerO is
    the ZerO initialisation of the same layer (the authors measure W - I).
    """
    residual = weight.detach() - reference.to(weight.device, weight.dtype)
    return {
        "stable_rank": stable_rank(weight),
        "rank": matrix_rank(weight),
        "residual_stable_rank": stable_rank(residual),
        "residual_rank": matrix_rank(residual),
        "max_rank": min(as_matrix(weight).shape),
    }


@torch.no_grad()
def zero_reference(model: nn.Module) -> dict[str, torch.Tensor]:
    """ZerO-initialised conv weights of the same architecture, keyed by module name."""
    ref_model = apply_zero_init(copy.deepcopy(model).cpu())
    return {
        name: m.weight.detach().clone()
        for name, m in ref_model.named_modules()
        if isinstance(m, nn.Conv2d)
    }


def tracked_convs(model: nn.Module) -> dict[str, nn.Conv2d]:
    """First convolution in the 2nd, 3rd and 4th groups of residual blocks (Fig. 5)."""
    return {
        f"{group}.0.conv1": getattr(model, group)[0].conv1
        for group in ("layer2", "layer3", "layer4")
    }


@torch.no_grad()
def tracked_rank_stats(
    model: nn.Module, reference: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    return {
        name: rank_stats(conv.weight, reference[name])
        for name, conv in tracked_convs(model).items()
    }


@torch.no_grad()
def kernel_ranks(
    model: nn.Module, reference: dict[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    """Rank statistics of every convolution kernel (Fig. 6, left)."""
    return {
        name: rank_stats(m.weight, reference[name])
        for name, m in model.named_modules()
        if isinstance(m, nn.Conv2d)
    }

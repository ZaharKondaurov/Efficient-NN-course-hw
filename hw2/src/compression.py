"""Model-complexity probes from Fig. 6: magnitude pruning and Tucker-2."""

from __future__ import annotations

import copy
from typing import Any, Callable

import torch
import torch.nn as nn

Evaluator = Callable[[nn.Module], dict[str, float]]


def prunable_weights(model: nn.Module) -> list[torch.Tensor]:
    return [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


@torch.no_grad()
def magnitude_prune_(model: nn.Module, sparsity: float) -> float:
    """
    Layer-wise unstructured magnitude pruning: in every Conv2d / Linear layer
    zero the ``sparsity`` fraction of weights with the smallest |w|.
    Returns the achieved global sparsity.
    """
    zeros, total = 0, 0
    for w in prunable_weights(model):
        k = int(round(sparsity * w.numel()))
        if k > 0:
            flat = w.abs().flatten()
            idx = torch.topk(flat, k, largest=False).indices
            w.view(-1)[idx] = 0
        zeros += int((w == 0).sum().item())
        total += w.numel()
    return zeros / max(total, 1)


def pruning_curve(model: nn.Module, sparsities: list[float], evaluate: Evaluator) -> list[dict]:
    rows = []
    for s in sparsities:
        pruned = copy.deepcopy(model)
        achieved = magnitude_prune_(pruned, s)
        rows.append({"sparsity": s, "achieved_sparsity": achieved, **evaluate(pruned)})
        print(f"  prune sparsity={s:.2f} -> err={rows[-1]['top1_error']:.4f}")
    return rows


def _leading_left_singular(mat: torch.Tensor, rank: int) -> torch.Tensor:
    u, _, _ = torch.linalg.svd(mat, full_matrices=False)
    return u[:, :rank]


@torch.no_grad()
def tucker2(
    weight: torch.Tensor, rank_out: int, rank_in: int, n_iter: int = 5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tucker-2 over the channel modes of a (cout, cin, k, k) kernel:
    W ≈ core ×_0 U_out ×_1 U_in with core (r_out, r_in, k, k).
    HOSVD initialisation refined by ``n_iter`` HOOI sweeps.
    """
    w = weight.detach().float()
    cout, cin, kh, kw = w.shape
    u_in = _leading_left_singular(w.transpose(0, 1).reshape(cin, -1), rank_in)
    u_out = _leading_left_singular(w.reshape(cout, -1), rank_out)

    for _ in range(n_iter):
        y = torch.einsum("oihw,is->oshw", w, u_in)
        u_out = _leading_left_singular(y.reshape(cout, -1), rank_out)
        y = torch.einsum("oihw,or->rihw", w, u_out)
        u_in = _leading_left_singular(y.transpose(0, 1).reshape(cin, -1), rank_in)

    core = torch.einsum("oihw,or,is->rshw", w, u_out, u_in)
    return core, u_out, u_in


def tucker2_reconstruct(core: torch.Tensor, u_out: torch.Tensor, u_in: torch.Tensor) -> torch.Tensor:
    return torch.einsum("rshw,or,is->oihw", core, u_out, u_in)


def tucker2_param_ratio(cout: int, cin: int, k: int, rank_out: int, rank_in: int) -> float:
    """Params of the 1x1 -> kxk -> 1x1 factorisation over the original kernel."""
    return (cin * rank_in + rank_in * rank_out * k * k + rank_out * cout) / (cout * cin * k * k)


def tucker_curve(
    model: nn.Module, layer_name: str, ranks: list[int], evaluate: Evaluator
) -> dict[str, Any]:
    conv = dict(model.named_modules())[layer_name]
    if not isinstance(conv, nn.Conv2d):
        raise ValueError(f"{layer_name} is not a Conv2d")
    cout, cin, k, _ = conv.weight.shape
    rows = []
    for r in ranks:
        r_out, r_in = min(r, cout), min(r, cin)
        approx = copy.deepcopy(model)
        target = dict(approx.named_modules())[layer_name]
        core, u_out, u_in = tucker2(target.weight, r_out, r_in)
        recon = tucker2_reconstruct(core, u_out, u_in)
        rel_err = ((recon - target.weight.float()).norm() / target.weight.float().norm()).item()
        target.weight.data.copy_(recon.to(target.weight))
        rows.append(
            {
                "rank": r,
                "param_ratio": tucker2_param_ratio(cout, cin, k, r_out, r_in),
                "relative_error": rel_err,
                **evaluate(approx),
            }
        )
        print(f"  tucker2 rank={r} -> err={rows[-1]['top1_error']:.4f} rel_err={rel_err:.3f}")
    return {"layer": layer_name, "shape": [cout, cin, k, k], "curve": rows}

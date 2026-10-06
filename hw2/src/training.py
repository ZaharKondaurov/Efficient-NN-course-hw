"""CNN training for the ZerO experiments (Tables 2–3, Figs. 4–6)."""

from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .data import DATASETS
from .inits import init_cnn
from .metrics import kernel_ranks, tracked_rank_stats, zero_reference
from .resnet import resnet

# Paper setup: ResNet-18 on CIFAR-10, ResNet-50 on ImageNet; lr 0.1, momentum 0.9,
# wd 1e-4, warmup 10 / 5 epochs. Epochs and decay milestones are not given in the
# paper and follow the standard ResNet recipes.
PAPER_HPARAMS: dict[str, dict[str, Any]] = {
    "cifar10": {
        "depth": 18, "epochs": 100, "batch_size": 256, "lr": 0.1, "momentum": 0.9,
        "weight_decay": 1e-4, "warmup_epochs": 10, "milestones": (50, 75),
    },
    "imagenet": {
        "depth": 50, "epochs": 60, "batch_size": 256, "lr": 0.1, "momentum": 0.9,
        "weight_decay": 1e-4, "warmup_epochs": 5, "milestones": (20, 40, 50),
    },
}


@dataclass
class TrainConfig:
    """Defaults follow the paper (lr 0.1, momentum 0.9, wd 1e-4, 10 warmup epochs)."""

    init: str = "zero"  # "zero" | "kaiming" | "xavier" | "rezero"
    dataset: str = "cifar10"  # "cifar10" | "imagenet"
    depth: int = 18
    block: str | None = None  # None -> standard block for depth; "basic" | "bottleneck"
    norm: str = "bn"  # "bn" | "none" (learnable scalar multiplier + bias)
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_epochs: int = 10  # paper: 10 for CIFAR-10, 5 for ImageNet
    milestones: tuple[int, ...] = (100, 150)
    gamma: float = 0.1
    seed: int = 0
    data_dir: str = "hw2/data"
    out_dir: str = "hw2/results"
    model_dir: str | None = None  # checkpoints go here; defaults to out_dir
    num_workers: int = 4
    rank_log_interval: int = 100  # iterations between stable-rank logs; 0 disables
    amp: bool = False
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def tag(self) -> str:
        block = f"-{self.block}" if self.block else ""
        return f"{self.dataset}_resnet{self.depth}{block}_{self.norm}_{self.init}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: TrainConfig) -> nn.Module:
    spec = DATASETS[cfg.dataset]
    model = resnet(
        cfg.depth,
        num_classes=spec["num_classes"],
        stem=spec["stem"],
        norm=cfg.norm,
        block=cfg.block,
        rezero=(cfg.init == "rezero"),
    )
    init_cnn(model, cfg.init)
    return model.to(cfg.device)


def config_from_dict(raw: dict[str, Any]) -> TrainConfig:
    raw = dict(raw)
    raw["milestones"] = tuple(raw.get("milestones", ()))
    return TrainConfig(**raw)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> SGD:
    return SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(optimizer: SGD, cfg: TrainConfig, steps_per_epoch: int) -> LambdaLR:
    """Per-iteration linear warmup, then step decay at ``milestones`` (in epochs)."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def factor(step: int) -> float:
        warm = min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0
        epoch = step // steps_per_epoch
        return warm * cfg.gamma ** sum(epoch >= m for m in cfg.milestones)

    return LambdaLR(optimizer, lr_lambda=factor)


def _autocast(cfg: TrainConfig):
    if cfg.amp and str(cfg.device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images = images.to(cfg.device, non_blocking=True)
        targets = targets.to(cfg.device, non_blocking=True)
        with _autocast(cfg):
            logits = model(images)
        total_loss += criterion(logits.float(), targets).item()
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    acc = correct / max(total, 1)
    return {"loss": total_loss / max(total, 1), "top1_acc": acc, "top1_error": 1.0 - acc}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: SGD,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    on_step: Callable[[], None] | None = None,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0

    for images, targets in loader:
        images = images.to(cfg.device, non_blocking=True)
        targets = targets.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(cfg):
            logits = model(images)
            loss = criterion(logits.float(), targets)
        if not math.isfinite(loss.item()):
            return {"loss": float("nan"), "top1_error": float("nan"), "diverged": True}
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_sum += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
        if on_step is not None:
            on_step()

    return {
        "loss": loss_sum / max(total, 1),
        "top1_error": 1.0 - correct / max(total, 1),
        "diverged": False,
    }


def save_checkpoint(
    path: Path, model: nn.Module, epoch: int, metrics: dict[str, Any], cfg: TrainConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "metrics": metrics, "config": asdict(cfg)},
        path,
    )


def fit(cfg: TrainConfig, train_loader: DataLoader, test_loader: DataLoader) -> dict[str, Any]:
    """One training run; writes ``run_<tag>_seed<N>.json`` and checkpoints."""
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cfg.tag}_seed{cfg.seed}"

    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    reference = zero_reference(model)

    step = 0
    rank_trajectory: list[dict[str, Any]] = []

    def log_ranks(epoch: int) -> None:
        rank_trajectory.append(
            {"iter": step, "epoch": epoch, "convs": tracked_rank_stats(model, reference)}
        )

    if cfg.rank_log_interval:
        log_ranks(0)

    history: list[dict[str, Any]] = []
    diverged = False

    for epoch in range(cfg.epochs):
        t0 = time.time()

        def on_step() -> None:
            nonlocal step
            step += 1
            if cfg.rank_log_interval and step % cfg.rank_log_interval == 0:
                log_ranks(epoch)

        train_stats = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, on_step)
        if train_stats["diverged"]:
            diverged = True
            print(f"[{run_name}] diverged at epoch {epoch}")
            break
        test_stats = evaluate(model, test_loader, cfg)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"],
            "train_top1_error": train_stats["top1_error"],
            "test_loss": test_stats["loss"],
            "test_top1_acc": test_stats["top1_acc"],
            "test_top1_error": test_stats["top1_error"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"[{run_name}] epoch {epoch:03d}/{cfg.epochs - 1} lr={row['lr']:.4g} "
            f"train_err={row['train_top1_error']:.4f} test_err={row['test_top1_error']:.4f} "
            f"test_acc={row['test_top1_acc']:.4f} ({row['seconds']:.1f}s)"
        )

    final = history[-1] if history else {}
    best = min(history, key=lambda r: r["test_top1_error"]) if history else {}
    if history:
        model_dir = Path(cfg.model_dir) if cfg.model_dir else out_dir
        save_checkpoint(model_dir / f"last_{run_name}.pt", model, final["epoch"], final, cfg)

    summary = {
        "config": asdict(cfg),
        "tag": cfg.tag,
        "diverged": diverged,
        "final": final,
        "best": best,
        "history": history,
        "rank_trajectory": rank_trajectory,
        "kernel_ranks": kernel_ranks(model, reference),
    }
    with open(out_dir / f"run_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

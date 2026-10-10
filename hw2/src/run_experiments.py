#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hw2.src.data import DATASETS
from hw2.src.inits import CNN_INITS
from hw2.src.metrics import mean_std
from hw2.src.report import (
    plot_kernel_rank_by_layer,
    plot_kernel_ranks_all_layers,
    plot_quality_metrics,
    plot_stable_rank_trajectories,
    plot_training_curves,
)
from hw2.src.training import PAPER_HPARAMS, TrainConfig, fit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="k experiments with ZerO / Kaiming / Xavier using the paper's hyperparameters"
    )
    p.add_argument("--dataset", choices=tuple(DATASETS), default="cifar10")
    p.add_argument("-k", "--k", type=int, default=10, help="number of seeds per init")
    p.add_argument("--inits", nargs="+", choices=CNN_INITS, default=list(CNN_INITS))
    p.add_argument("--depth", type=int, default=None,
                   help="defaults to the paper: 18 for CIFAR-10, 50 for ImageNet")
    p.add_argument("--norm", choices=("bn", "none"), default="bn",
                   help="bn = BatchNorm; none = learnable scalar scale/bias (no BN)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--milestones", nargs="+", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--exp-dir", type=Path, default=None,
                   help="defaults to hw2/experiments/<dataset>_resnet<depth>")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--rank-log-interval", type=int, default=100)
    p.add_argument("--amp", action="store_true", help="bf16 autocast")
    p.add_argument("--device", default=None)
    p.add_argument("--plot-only", action="store_true",
                   help="only rebuild figures from <exp-dir>/metrics.json")
    return p.parse_args()


def summarize(runs: list[dict]) -> dict[str, dict]:
    ok = [r for r in runs if r["final"] and not r["diverged"]]
    if not ok:
        return {"n_runs": len(runs), "n_diverged": len(runs)}
    pct = lambda key, part: [100 * r[part][key] for r in ok]  # noqa: E731
    return {
        "n_runs": len(runs),
        "n_diverged": len(runs) - len(ok),
        "test_top1_error_pct": mean_std(pct("test_top1_error", "final")),
        "test_top1_acc_pct": mean_std(pct("test_top1_acc", "final")),
        "best_test_top1_error_pct": mean_std(pct("test_top1_error", "best")),
        "best_test_top1_acc_pct": mean_std(pct("test_top1_acc", "best")),
        "train_top1_error_pct": mean_std(pct("train_top1_error", "final")),
    }


def summary_table(title: str, summary: dict[str, dict]) -> str:
    fmt = lambda s: f"{s['mean']:.2f} ± {s['std']:.2f}"  # noqa: E731
    lines = [
        f"## {title}",
        "",
        "| Init | Runs | Diverged | Top-1 test error, % | Accuracy, % | Best-epoch error, % |",
        "|---|---|---|---|---|---|",
    ]
    for init, s in summary.items():
        if "test_top1_error_pct" not in s:
            lines.append(f"| {init} | {s['n_runs']} | {s['n_diverged']} | — | — | — |")
            continue
        lines.append(
            f"| {init} | {s['n_runs']} | {s['n_diverged']} | {fmt(s['test_top1_error_pct'])} "
            f"| {fmt(s['test_top1_acc_pct'])} | {fmt(s['best_test_top1_error_pct'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    hp = dict(PAPER_HPARAMS[args.dataset])
    depth = args.depth or hp["depth"]
    del hp["depth"]
    if args.epochs is not None:
        hp["epochs"] = args.epochs
        hp["milestones"] = tuple(m for m in hp["milestones"] if m < args.epochs)
        hp["warmup_epochs"] = min(hp["warmup_epochs"], max(args.epochs - 1, 0))

    exp_dir = args.exp_dir or Path("hw2/experiments") / (
        f"{args.dataset}_resnet{depth}" + ("_nobn" if args.norm == "none" else "")
    )
    if args.plot_only:
        plot_only(exp_dir)
        return
    model_dir, runs_dir = exp_dir / "models", exp_dir / "runs"
    model_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or ("hw2/data" if args.dataset == "cifar10" else "hw2/data/imagenet")

    train_loader, test_loader = DATASETS[args.dataset]["loaders"](
        data_dir, batch_size=hp["batch_size"], num_workers=args.num_workers
    )

    seeds = list(range(args.k))
    per_init: dict[str, list[dict]] = {}
    configs: dict[str, dict] = {}
    t0 = time.time()
    for init in args.inits:
        per_init[init] = []
        for seed in seeds:
            cfg = TrainConfig(
                init=init,
                dataset=args.dataset,
                depth=depth,
                norm=args.norm,
                seed=seed,
                data_dir=data_dir,
                out_dir=str(runs_dir),
                model_dir=str(model_dir),
                num_workers=args.num_workers,
                rank_log_interval=args.rank_log_interval,
                amp=args.amp,
                **hp,
                **({"device": args.device} if args.device else {}),
            )
            per_init[init].append(fit(cfg, train_loader, test_loader))
        configs[init] = {k: v for k, v in asdict(cfg).items() if k != "seed"}

    title = f"{args.dataset} ResNet-{depth} ({args.norm}), {args.k} seeds"
    summary = {init: summarize(rs) for init, rs in per_init.items()}
    metrics = {
        "dataset": args.dataset,
        "depth": depth,
        "seeds": seeds,
        "hyperparameters": configs,
        "wall_time_s": time.time() - t0,
        "summary": summary,
        "runs": {
            init: [
                {
                    "seed": r["config"]["seed"],
                    "diverged": r["diverged"],
                    "final": r["final"],
                    "best": r["best"],
                    "history": r["history"],
                    "kernel_ranks": r["kernel_ranks"],
                    "rank_trajectory": r["rank_trajectory"],
                }
                for r in rs
            ]
            for init, rs in per_init.items()
        },
    }
    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    table = summary_table(title, summary)
    (exp_dir / "summary.md").write_text(table, encoding="utf-8")
    make_figures(metrics["runs"], exp_dir, title)

    print(table)
    print(f"results -> {exp_dir}")


def make_figures(per_init: dict[str, list[dict]], exp_dir: Path, title: str) -> None:
    plot_training_curves(per_init, exp_dir / "training_curves.png", title)
    plot_quality_metrics(per_init, exp_dir / "quality_metrics.png", title)
    plot_stable_rank_trajectories(per_init, exp_dir / "stable_rank.png", title)
    plot_kernel_rank_by_layer(per_init, exp_dir / "kernel_rank_vs_layer.png", title, key="rank")
    plot_kernel_ranks_all_layers(per_init, exp_dir / "kernel_ranks.png", title)


def plot_only(exp_dir: Path) -> None:
    metrics = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
    title = f"{metrics['dataset']} ResNet-{metrics['depth']}, {len(metrics['seeds'])} seeds"
    make_figures(metrics["runs"], exp_dir, title)
    print(f"figures -> {exp_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI: train ResNets with ZerO / Kaiming / Xavier over several seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hw2.src.data import DATASETS
from hw2.src.inits import CNN_INITS
from hw2.src.metrics import mean_std
from hw2.src.training import PAPER_HPARAMS, TrainConfig, fit

DATASET_DEFAULTS = PAPER_HPARAMS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZerO reproduction: ResNet on CIFAR-10 / ImageNet")
    p.add_argument("--inits", nargs="+", choices=CNN_INITS, default=["zero"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--dataset", choices=tuple(DATASETS), default="cifar10")
    p.add_argument("--depth", type=int, default=18, help="18/34/50/101/152 or 8n+2 / 12n+2")
    p.add_argument("--block", choices=("basic", "bottleneck"), default=None)
    p.add_argument("--norm", choices=("bn", "none"), default="bn")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--milestones", type=int, nargs="*", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default="hw2/results")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--rank-log-interval", type=int, default=100,
                   help="iterations between stable-rank logs of tracked convs (0 disables)")
    p.add_argument("--amp", action="store_true", help="bf16 autocast")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    pick = lambda value, key: defaults[key] if value is None else value  # noqa: E731

    epochs = pick(args.epochs, "epochs")
    batch_size = pick(args.batch_size, "batch_size")
    data_dir = args.data_dir or ("hw2/data" if args.dataset == "cifar10" else "hw2/data/imagenet")

    loader_kwargs = {"batch_size": batch_size, "num_workers": args.num_workers}
    train_loader, test_loader = DATASETS[args.dataset]["loaders"](data_dir, **loader_kwargs)

    for init in args.inits:
        errors, accs = [], []
        for seed in args.seeds:
            cfg = TrainConfig(
                init=init,
                dataset=args.dataset,
                depth=args.depth,
                block=args.block,
                norm=args.norm,
                epochs=epochs,
                batch_size=batch_size,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                warmup_epochs=pick(args.warmup_epochs, "warmup_epochs"),
                milestones=tuple(pick(args.milestones, "milestones")),
                seed=seed,
                data_dir=data_dir,
                out_dir=args.out_dir,
                num_workers=args.num_workers,
                rank_log_interval=args.rank_log_interval,
                amp=args.amp,
                **({"device": args.device} if args.device else {}),
            )
            summary = fit(cfg, train_loader, test_loader)
            if summary["final"]:
                errors.append(summary["final"]["test_top1_error"])
                accs.append(summary["final"]["test_top1_acc"])

        if errors:
            err, acc = mean_std([100 * e for e in errors]), mean_std([100 * a for a in accs])
            print(
                f"== {cfg.tag}: top-1 test error {err['mean']:.2f} ± {err['std']:.2f} %, "
                f"accuracy {acc['mean']:.2f} ± {acc['std']:.2f} % over {err['n']} seeds"
            )


if __name__ == "__main__":
    main()

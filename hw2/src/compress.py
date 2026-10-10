#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hw2.src.compression import pruning_curve, tucker_curve
from hw2.src.data import DATASETS
from hw2.src.report import plot_compression
from hw2.src.training import build_model, config_from_dict, evaluate

DEFAULT_SPARSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
DEFAULT_RANKS = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pruning / Tucker-2 on trained checkpoints")
    p.add_argument("checkpoints", nargs="+", type=Path, help="last_*.pt files")
    p.add_argument("--sparsities", nargs="+", type=float, default=DEFAULT_SPARSITIES)
    p.add_argument("--tucker-layer", default="layer4.1.conv1", help="a 512-channel conv")
    p.add_argument("--ranks", nargs="+", type=int, default=DEFAULT_RANKS)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write compression_*.json and the plot (default: experiment root)",
    )
    return p.parse_args()


def _exp_dir(ckpt_path: Path) -> Path:
    """last_*.pt lives in <exp>/models/ → write plots/json to <exp>/."""
    return ckpt_path.parent.parent if ckpt_path.parent.name == "models" else ckpt_path.parent


def main() -> None:
    args = parse_args()
    loaders: dict[tuple[str, str], object] = {}
    by_out: dict[Path, list[dict]] = defaultdict(list)

    for ckpt_path in args.checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = config_from_dict(ckpt["config"])
        cfg.device = args.device
        model = build_model(cfg)
        model.load_state_dict(ckpt["model"])

        key = (cfg.dataset, cfg.data_dir)
        if key not in loaders:
            _, loaders[key] = DATASETS[cfg.dataset]["loaders"](
                cfg.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
            )
        test_loader = loaders[key]
        evaluator = lambda m, _cfg=cfg, _loader=test_loader: evaluate(m, _loader, _cfg)

        print(f"{ckpt_path.name}: magnitude pruning")
        pruning = pruning_curve(model, args.sparsities, evaluator)
        print(f"{ckpt_path.name}: Tucker-2 on {args.tucker_layer}")
        tucker = tucker_curve(model, args.tucker_layer, args.ranks, evaluator)

        payload = {
            "checkpoint": str(ckpt_path),
            "tag": cfg.tag,
            "seed": cfg.seed,
            "pruning": pruning,
            "tucker2": tucker,
        }
        out_dir = args.out_dir or _exp_dir(ckpt_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (ckpt_path.stem.replace("last_", "compression_") + ".json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved {out}")
        by_out[out_dir].append(payload)

    for out_dir, results in by_out.items():
        plot_compression(results, out_dir)
        print(f"plot -> {out_dir}/compression_*.png")


if __name__ == "__main__":
    main()

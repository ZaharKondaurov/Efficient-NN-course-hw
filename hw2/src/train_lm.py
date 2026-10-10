#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hw2.src.data import Corpus
from hw2.src.inits import TRANSFORMER_INITS
from hw2.src.lm_training import LMConfig, fit_lm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZerO reproduction: Transformer LM on WikiText-2")
    p.add_argument("--inits", nargs="+", choices=TRANSFORMER_INITS, default=["standard", "zero"])
    p.add_argument("--layers", nargs="+", type=int, default=[2, 4, 6, 8, 10, 20])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--decay-epoch", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.25)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--d-model", type=int, default=200)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--d-hid", type=int, default=200)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--bptt", type=int, default=35)
    p.add_argument("--data-dir", default="hw2/data")
    p.add_argument("--out-dir", default="hw2/results")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus = Corpus(args.data_dir)
    print(f"WikiText-2: vocab={corpus.vocab_size} train_tokens={corpus.train.numel()}")

    for num_layers in args.layers:
        for init in args.inits:
            for seed in args.seeds:
                cfg = LMConfig(
                    init=init,
                    num_layers=num_layers,
                    d_model=args.d_model,
                    nhead=args.nhead,
                    d_hid=args.d_hid,
                    dropout=args.dropout,
                    epochs=args.epochs,
                    lr=args.lr,
                    decay_epoch=args.decay_epoch,
                    gamma=args.gamma,
                    warmup_epochs=args.warmup_epochs,
                    batch_size=args.batch_size,
                    bptt=args.bptt,
                    seed=seed,
                    data_dir=args.data_dir,
                    out_dir=args.out_dir,
                    **({"device": args.device} if args.device else {}),
                )
                fit_lm(cfg, corpus)


if __name__ == "__main__":
    main()

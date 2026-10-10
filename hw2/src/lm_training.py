from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .data import Corpus, batchify, get_batch
from .metrics import perplexity
from .training import set_seed
from .transformer import TransformerLM


@dataclass
class LMConfig:
    """PyTorch word_language_model defaults; 20 epochs with one LR decay at epoch 10."""

    init: str = "zero"  # "zero" | "standard" | "kaiming" | "xavier" | "rezero"
    num_layers: int = 2
    d_model: int = 200
    nhead: int = 2
    d_hid: int = 200
    dropout: float = 0.2
    epochs: int = 20
    lr: float = 0.1
    decay_epoch: int = 10
    gamma: float = 0.25
    clip: float = 0.25
    warmup_epochs: int = 0
    batch_size: int = 20
    eval_batch_size: int = 10
    bptt: int = 35
    seed: int = 0
    data_dir: str = "hw2/data"
    out_dir: str = "hw2/results"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    @property
    def tag(self) -> str:
        return f"wikitext2_transformer_L{self.num_layers}_{self.init}"


@torch.no_grad()
def evaluate_lm(model: nn.Module, source: torch.Tensor, cfg: LMConfig, vocab: int) -> float:
    """Mean token negative log-likelihood over ``source``."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0
    for i in range(0, source.size(0) - 1, cfg.bptt):
        data, targets = get_batch(source, i, cfg.bptt)
        logits = model(data)
        total_loss += criterion(logits.view(-1, vocab), targets).item()
        total_tokens += targets.numel()
    return total_loss / max(total_tokens, 1)


def _lr_at(cfg: LMConfig, epoch: int, batch: int, batches_per_epoch: int) -> float:
    lr = cfg.lr * (cfg.gamma if epoch >= cfg.decay_epoch else 1.0)
    warmup_steps = cfg.warmup_epochs * batches_per_epoch
    step = epoch * batches_per_epoch + batch
    if warmup_steps > 0 and step < warmup_steps:
        lr *= (step + 1) / warmup_steps
    return lr


def fit_lm(cfg: LMConfig, corpus: Corpus) -> dict[str, Any]:
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{cfg.tag}_seed{cfg.seed}"

    vocab = corpus.vocab_size
    train_data = batchify(corpus.train, cfg.batch_size).to(cfg.device)
    val_data = batchify(corpus.valid, cfg.eval_batch_size).to(cfg.device)
    test_data = batchify(corpus.test, cfg.eval_batch_size).to(cfg.device)

    model = TransformerLM(
        vocab,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        d_hid=cfg.d_hid,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        init=cfg.init,
    ).to(cfg.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    batches_per_epoch = math.ceil((train_data.size(0) - 1) / cfg.bptt)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    diverged = False

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        loss_sum, tokens = 0.0, 0
        for batch, i in enumerate(range(0, train_data.size(0) - 1, cfg.bptt)):
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(cfg, epoch, batch, batches_per_epoch)
            data, targets = get_batch(train_data, i, cfg.bptt)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data).view(-1, vocab), targets)
            if not math.isfinite(loss.item()):
                diverged = True
                break
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            loss_sum += loss.item() * targets.numel()
            tokens += targets.numel()

        if diverged:
            print(f"[{run_name}] diverged at epoch {epoch}")
            break

        val_nll = evaluate_lm(model, val_data, cfg, vocab)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_ppl": perplexity(loss_sum / max(tokens, 1)),
            "val_ppl": perplexity(val_nll),
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(
            f"[{run_name}] epoch {epoch:02d} lr={row['lr']:.3g} "
            f"train_ppl={row['train_ppl']:.2f} val_ppl={row['val_ppl']:.2f} "
            f"({row['seconds']:.1f}s)"
        )
        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    test_ppl = float("nan")
    if best_state is not None:
        model.load_state_dict(best_state)
        test_ppl = perplexity(evaluate_lm(model, test_data, cfg, vocab))
    print(f"[{run_name}] test_ppl={test_ppl:.2f} diverged={diverged}")

    summary = {
        "config": asdict(cfg),
        "tag": cfg.tag,
        "diverged": diverged,
        "best_val_ppl": perplexity(best_val) if best_state is not None else float("nan"),
        "test_ppl": test_ppl,
        "history": history,
    }
    with open(out_dir / f"lm_{run_name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

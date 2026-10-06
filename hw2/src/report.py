#!/usr/bin/env python3
"""Aggregate results into paper-style tables (mean ± std) and figures."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hw2.src.metrics import mean_std  # noqa: E402


def _load(paths: list[Path]) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(paths)]


def _group_and_init(tag: str) -> tuple[str, str]:
    group, init = tag.rsplit("_", 1)
    return group, init


def _fmt(stats: dict[str, float]) -> str:
    return f"{stats['mean']:.2f} ± {stats['std']:.2f}"


def cnn_table(runs: list[dict]) -> str:
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_tag[r["tag"]].append(r)

    lines = [
        "| Setting | Init | Seeds | Diverged | Top-1 test error, % (mean ± std) "
        "| Accuracy, % (mean ± std) | Best-epoch error, % |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag in sorted(by_tag):
        group, init = _group_and_init(tag)
        ok = [r for r in by_tag[tag] if r["final"] and not r["diverged"]]
        diverged = len(by_tag[tag]) - len(ok)
        if not ok:
            lines.append(f"| {group} | {init} | {len(by_tag[tag])} | {diverged} | — | — | — |")
            continue
        err = mean_std([100 * r["final"]["test_top1_error"] for r in ok])
        acc = mean_std([100 * r["final"]["test_top1_acc"] for r in ok])
        best = mean_std([100 * r["best"]["test_top1_error"] for r in ok])
        lines.append(
            f"| {group} | {init} | {len(by_tag[tag])} | {diverged} | {_fmt(err)} "
            f"| {_fmt(acc)} | {_fmt(best)} |"
        )
    return "\n".join(lines)


def lm_table(runs: list[dict]) -> str:
    cells: dict[tuple[str, int], list[float]] = defaultdict(list)
    diverged: dict[tuple[str, int], int] = defaultdict(int)
    for r in runs:
        key = (r["config"]["init"], r["config"]["num_layers"])
        if r["diverged"] or not math.isfinite(r["test_ppl"]):
            diverged[key] += 1
        else:
            cells[key].append(r["test_ppl"])

    inits = sorted({k[0] for k in list(cells) + list(diverged)})
    layers = sorted({k[1] for k in list(cells) + list(diverged)})
    lines = [
        "| Init \\ layers | " + " | ".join(str(n) for n in layers) + " |",
        "|---" * (len(layers) + 1) + "|",
    ]
    for init in inits:
        row = []
        for n in layers:
            vals = cells.get((init, n), [])
            if vals:
                s = mean_std(vals)
                row.append(f"{s['mean']:.2f}" + (f" ± {s['std']:.2f}" if s["n"] > 1 else ""))
            else:
                row.append("diverged" if diverged.get((init, n)) else "—")
        lines.append(f"| {init} | " + " | ".join(row) + " |")
    return "\n".join(lines)


STABLE_RANK_ROWS = [
    ("residual_stable_rank", "stable rank of W - W_ZerO"),
    ("stable_rank", "stable rank of W"),
]


def plot_stable_rank_trajectories(
    per_init: dict[str, list[dict]],
    path: Path,
    title: str,
    rows: list[tuple[str, str]] = STABLE_RANK_ROWS,
) -> None:
    """
    Fig. 5: stable rank of the first conv in the 2nd/3rd/4th residual groups vs.
    iteration, mean ± std over seeds per init. Rows: residual component and raw W.
    """
    per_init = {i: [r for r in rs if r.get("rank_trajectory")] for i, rs in sorted(per_init.items())}
    per_init = {i: rs for i, rs in per_init.items() if rs}
    if not per_init:
        return
    first_run = next(iter(per_init.values()))[0]
    convs = list(first_run["rank_trajectory"][0]["convs"])

    fig, axes = plt.subplots(len(rows), len(convs), figsize=(4.8 * len(convs), 3.6 * len(rows)),
                             squeeze=False)
    for row, (key, label) in enumerate(rows):
        for col, conv in enumerate(convs):
            ax = axes[row, col]
            for k, (init, rs) in enumerate(per_init.items()):
                n = min(len(r["rank_trajectory"]) for r in rs)
                its = np.array([p["iter"] for p in rs[0]["rank_trajectory"][:n]])
                vals = np.array([[p["convs"][conv][key] for p in r["rank_trajectory"][:n]]
                                 for r in rs])
                _band(ax, its, vals, init, color=f"C{k}")
            max_rank = first_run["rank_trajectory"][0]["convs"][conv]["max_rank"]
            ax.axhline(max_rank, color="k", ls="--", lw=1, label="max rank")
            ax.set_title(f"{conv} ({label})", fontsize=10)
            ax.set_xlabel("iteration")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(f"{title}: stable rank during training")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stable_ranks(runs: list[dict], out_dir: Path) -> None:
    by_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        group, init = _group_and_init(r["tag"])
        by_group[group][init].append(r)
    for group, per_init in by_group.items():
        plot_stable_rank_trajectories(per_init, out_dir / f"stable_rank_{group}.png", group)


def plot_quality_metrics(per_init: dict[str, list[dict]], path: Path, title: str) -> None:
    """Top-1 test error and accuracy (final and best epoch), mean ± std over seeds."""
    per_init = {i: [r for r in rs if r["final"] and not r["diverged"]]
                for i, rs in sorted(per_init.items())}
    per_init = {i: rs for i, rs in per_init.items() if rs}
    if not per_init:
        return
    names = list(per_init)
    panels = [
        ("test_top1_error", "final", "Top-1 test error, % (final)"),
        ("test_top1_error", "best", "Top-1 test error, % (best epoch)"),
        ("test_top1_acc", "final", "Test accuracy, % (final)"),
        ("test_top1_acc", "best", "Test accuracy, % (best epoch)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4))
    for ax, (key, part, label) in zip(axes, panels):
        vals = [[100 * r[part][key] for r in per_init[i]] for i in names]
        means = np.array([np.mean(v) for v in vals])
        stds = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals])
        bars = ax.bar(names, means, yerr=stds, capsize=6,
                      color=[f"C{k}" for k in range(len(names))])
        for k, v in enumerate(vals):
            ax.scatter(np.full(len(v), k), v, color="k", s=10, zorder=3)
        for bar, m, s in zip(bars, means, stds):
            ax.annotate(f"{m:.2f} ± {s:.2f}", (bar.get_x() + bar.get_width() / 2, m + s),
                        ha="center", va="bottom", fontsize=9)
        lo = min(min(v) for v in vals)
        hi = max(max(v) for v in vals)
        pad = max(0.5, 0.3 * (hi - lo))
        ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + 2 * pad))
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    n = [len(per_init[i]) for i in names]
    fig.suptitle(f"{title}: quality metrics (mean ± std, seeds per init: {n})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


KERNEL_RANK_METRICS = [
    ("rank", "matrix rank of W"),
    ("stable_rank", "stable rank of W"),
    ("residual_rank", "matrix rank of W - W_ZerO"),
    ("residual_stable_rank", "stable rank of W - W_ZerO"),
]


def plot_kernel_ranks_all_layers(per_init: dict[str, list[dict]], path: Path, title: str) -> None:
    """Grouped bars over every conv layer (by name), mean ± std over seeds per init."""
    per_init = {i: [r for r in rs if r.get("kernel_ranks") and r["final"]]
                for i, rs in per_init.items()}
    per_init = {i: rs for i, rs in sorted(per_init.items()) if rs}
    if not per_init:
        return
    first = next(iter(per_init.values()))[0]["kernel_ranks"]
    layers = list(first)
    x = np.arange(len(layers))
    width = 0.8 / len(per_init)

    fig, axes = plt.subplots(len(KERNEL_RANK_METRICS), 1,
                             figsize=(max(12, 0.45 * len(layers)), 3.2 * len(KERNEL_RANK_METRICS)),
                             sharex=True)
    for ax, (key, label) in zip(axes, KERNEL_RANK_METRICS):
        for j, (init, rs) in enumerate(per_init.items()):
            vals = np.array([[r["kernel_ranks"][n][key] for n in layers] for r in rs])
            ax.bar(x + (j - (len(per_init) - 1) / 2) * width, vals.mean(0), width,
                   yerr=vals.std(0) if len(rs) > 1 else None, capsize=2, label=init)
        if key in ("rank", "residual_rank"):
            ax.step(x, [first[n]["max_rank"] for n in layers], "k--", lw=1, where="mid",
                    label="max rank")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(layers, rotation=70, ha="right", fontsize=8)
    fig.suptitle(f"{title}: kernel ranks of all conv layers after training")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kernel_ranks(runs: list[dict], out_dir: Path) -> None:
    by_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        group, init = _group_and_init(r["tag"])
        by_group[group][init].append(r)
    for group, per_init in by_group.items():
        plot_kernel_ranks_all_layers(per_init, out_dir / f"kernel_rank_{group}.png", group)


def _stack(rs: list[dict], key: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-epoch values over seeds, truncated to the shortest (e.g. diverged) run."""
    n = min(len(r["history"]) for r in rs)
    vals = np.array([[scale * row[key] for row in r["history"][:n]] for r in rs])
    return np.arange(1, n + 1), vals


def _band(ax, epochs: np.ndarray, vals: np.ndarray, label: str, **kw) -> None:
    mean = vals.mean(0)
    line, = ax.plot(epochs, mean, label=label, **kw)
    if len(vals) > 1:
        std = vals.std(0)
        ax.fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=0.2)


def plot_training_curves(per_init: dict[str, list[dict]], path: Path, title: str) -> None:
    """Loss, top-1 error, test accuracy per epoch (mean ± std over seeds) + final accuracy."""
    per_init = {i: [r for r in rs if r["history"]] for i, rs in sorted(per_init.items())}
    per_init = {i: rs for i, rs in per_init.items() if rs}
    if not per_init:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_loss, ax_err, ax_acc, ax_bar = axes.flat
    for k, (init, rs) in enumerate(per_init.items()):
        color = f"C{k}"
        e, v = _stack(rs, "train_loss")
        _band(ax_loss, e, v, f"{init} train", color=color, ls="--")
        e, v = _stack(rs, "test_loss")
        _band(ax_loss, e, v, f"{init} test", color=color)
        e, v = _stack(rs, "train_top1_error", 100)
        _band(ax_err, e, v, f"{init} train", color=color, ls="--")
        e, v = _stack(rs, "test_top1_error", 100)
        _band(ax_err, e, v, f"{init} test", color=color)
        e, v = _stack(rs, "test_top1_acc", 100)
        _band(ax_acc, e, v, init, color=color)

    ax_loss.set(title="Cross-entropy loss", xlabel="epoch", ylabel="loss", yscale="log")
    ax_err.set(title="Top-1 error", xlabel="epoch", ylabel="error, %", yscale="log")
    ax_acc.set(title="Test accuracy", xlabel="epoch", ylabel="accuracy, %")
    for ax in (ax_loss, ax_err, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    names = list(per_init)
    finals = [[100 * r["final"]["test_top1_acc"] for r in per_init[i] if r["final"]] for i in names]
    means = [np.mean(f) for f in finals]
    stds = [np.std(f, ddof=1) if len(f) > 1 else 0.0 for f in finals]
    bars = ax_bar.bar(names, means, yerr=stds, capsize=6,
                      color=[f"C{k}" for k in range(len(names))])
    for bar, m, s in zip(bars, means, stds):
        ax_bar.annotate(f"{m:.2f} ± {s:.2f}", (bar.get_x() + bar.get_width() / 2, m + s),
                        ha="center", va="bottom", fontsize=9)
    lo = min(m - s for m, s in zip(means, stds))
    ax_bar.set_ylim(max(0.0, lo - 2), min(100.0, max(means) + max(stds) + 2))
    ax_bar.set(title=f"Final test accuracy (n={len(finals[0])} seeds)", ylabel="accuracy, %")
    ax_bar.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_compression(results: list[dict], out_dir: Path) -> None:
    by_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        group, init = _group_and_init(r["tag"])
        by_group[group][init].append(r)

    for group, per_init in by_group.items():
        fig, (ax_p, ax_t) = plt.subplots(1, 2, figsize=(10, 3.8))
        for init, rs in sorted(per_init.items()):
            xs = [row["sparsity"] for row in rs[0]["pruning"]]
            ys = np.mean([[100 * row["top1_acc"] for row in r["pruning"]] for r in rs], 0)
            ax_p.plot(xs, ys, marker="o", label=init)

            ranks = [row["rank"] for row in rs[0]["tucker2"]["curve"]]
            ys = np.mean([[100 * row["top1_acc"] for row in r["tucker2"]["curve"]] for r in rs], 0)
            ax_t.plot(ranks, ys, marker="o", label=init)
        ax_p.set_title("Magnitude-based pruning")
        ax_p.set_xlabel("sparsity")
        ax_p.set_ylabel("test accuracy, %")
        ax_t.set_title(f"Tucker-2 on {per_init[next(iter(per_init))][0]['tucker2']['layer']}")
        ax_t.set_xlabel("Tucker rank (r_in = r_out)")
        ax_t.set_ylabel("test accuracy, %")
        ax_t.set_xscale("log", base=2)
        for ax in (ax_p, ax_t):
            ax.legend()
        fig.suptitle(group)
        fig.tight_layout()
        fig.savefig(out_dir / f"compression_{group}.png", dpi=150)
        plt.close(fig)


def plot_lm(runs: list[dict], out_dir: Path) -> None:
    per_init: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if not r["diverged"] and math.isfinite(r["test_ppl"]):
            per_init[r["config"]["init"]][r["config"]["num_layers"]].append(r["test_ppl"])
    if not per_init:
        return
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for init, by_layers in sorted(per_init.items()):
        layers = sorted(by_layers)
        ax.plot(layers, [np.mean(by_layers[n]) for n in layers], marker="o", label=init)
    ax.set_xlabel("number of Transformer layers")
    ax.set_ylabel("test perplexity")
    ax.set_title("WikiText-2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lm_perplexity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Build tables and figures from hw2 results")
    p.add_argument("--results-dir", type=Path, default=Path("hw2/results"))
    args = p.parse_args()

    res = args.results_dir
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cnn_runs = _load(list(res.glob("run_*.json")))
    lm_runs = _load(list(res.glob("lm_*.json")))
    compression = _load(list(res.glob("compression_*.json")))

    report = []
    if cnn_runs:
        report += ["## CNN: top-1 test error / accuracy (final epoch)", "", cnn_table(cnn_runs), ""]
        by_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for r in cnn_runs:
            group, init = _group_and_init(r["tag"])
            by_group[group][init].append(r)
        for group, per_init in by_group.items():
            plot_stable_rank_trajectories(per_init, fig_dir / f"stable_rank_{group}.png", group)
            plot_kernel_ranks_all_layers(per_init, fig_dir / f"kernel_rank_{group}.png", group)
            plot_quality_metrics(per_init, fig_dir / f"quality_metrics_{group}.png", group)
            plot_training_curves(per_init, fig_dir / f"training_curves_{group}.png", group)
    if lm_runs:
        report += ["## Transformer on WikiText-2: test perplexity", "", lm_table(lm_runs), ""]
        plot_lm(lm_runs, fig_dir)
    if compression:
        plot_compression(compression, fig_dir)

    text = "\n".join(report)
    (res / "tables.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynvml
import torch
from tqdm.auto import tqdm

from calibrate import measure_energy_j, measure_median_latency_s
from equations import arithmetic_intensity, bytes_moved, energy, flops, latency, memory
from models import Model

from torch.utils.flop_counter import FlopCounterMode


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
THETA_PATH = RESULTS_DIR / "theta.json"
MEASUREMENTS_PATH = RESULTS_DIR / "measurements.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"

WARMUP_ITERS = 5
LATENCY_REPEATS = 30
ENERGY_REPS = 50

BASE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {
            "n": 0,
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "mape": float("nan"),
        }

    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    nonzero = np.abs(y_true) > 1e-12
    mape = (
        float(np.mean(np.abs(err[nonzero] / y_true[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    return {
        "n": int(y_true.size),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def compute_metrics(df: pd.DataFrame) -> dict:
    """MSE / RMSE / MAE / MAPE for latency, energy, memory (all / train / val)."""
    ok = df.loc[df["oom"] == False].copy()
    metrics: dict = {
        "n_total": int(len(df)),
        "n_oom": int(df["oom"].sum()),
        "n_ok": int(len(ok)),
    }

    pairs = [
        ("latency_s", "real_latency_s", "pred_latency_s"),
        ("energy_j", "real_energy_j", "pred_energy_j"),
        ("memory_bytes", "real_memory_bytes", "pred_memory_bytes"),
    ]
    if "flops_torch" in ok.columns:
        pairs.append(("flops", "flops_torch", "flops"))
    splits = {
        "all": ok,
        "train": ok.loc[~ok["is_validation"]],
        "validation": ok.loc[ok["is_validation"]],
    }

    for split_name, split_df in splits.items():
        metrics[split_name] = {}
        for metric_name, real_col, pred_col in pairs:
            metrics[split_name][metric_name] = _regression_stats(
                split_df[real_col].to_numpy(),
                split_df[pred_col].to_numpy(),
            )
    return metrics


def save_metrics(df: pd.DataFrame, path: Path = METRICS_PATH) -> dict:
    metrics = compute_metrics(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def measure_peak_memory_bytes(model: Model, x: torch.Tensor) -> int:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model(x)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def measure_torch_flops(model: Model, x: torch.Tensor) -> float:
    """FLOPs of one forward via torch.utils.flop_counter.FlopCounterMode."""
    if FlopCounterMode is None:
        raise RuntimeError("torch.utils.flop_counter.FlopCounterMode is unavailable")
    with FlopCounterMode(display=False) as flop_counter:
        model(x)
    return float(flop_counter.get_total_flops())


def build_grid() -> tuple[list[int], list[int], set[tuple[int, int]]]:
    image_sizes = list(BASE_SIZES)
    batches = list(BASE_BATCHES)

    extra_sizes: list[int] = []
    while len(extra_sizes) < 4:
        s = 16 * random.randint(2, 32)
        if s not in BASE_SIZES and s not in extra_sizes:
            extra_sizes.append(s)
    image_sizes.extend(extra_sizes)

    powers = set(BASE_BATCHES)
    extra_batches: list[int] = []
    while len(extra_batches) < 3:
        b = random.randint(1, 256)
        if b not in powers and b not in extra_batches:
            extra_batches.append(b)
    batches.extend(extra_batches)

    validation = {(s, b) for s in extra_sizes for b in batches} | {
        (s, b) for s in image_sizes for b in extra_batches
    }
    return image_sizes, batches, validation


def load_thetas(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    latency_theta = np.array(
        [
            payload["theta_latency"]["theta_0"],
            payload["theta_latency"]["theta_flops"],
            payload["theta_latency"]["theta_bytes"],
        ],
        dtype=float,
    )
    energy_theta = np.array(
        [
            payload["theta_energy"]["theta_static"],
            payload["theta_energy"]["theta_flops"],
            payload["theta_energy"]["theta_bytes"],
        ],
        dtype=float,
    )
    return latency_theta, energy_theta


def _scatter_with_identity(
    ax: plt.Axes,
    measured: np.ndarray,
    predicted: np.ndarray,
    is_validation: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    train = ~is_validation
    if train.any():
        ax.scatter(
            measured[train],
            predicted[train],
            s=28,
            alpha=0.75,
            label="train / base grid",
        )
    if is_validation.any():
        ax.scatter(
            measured[is_validation],
            predicted[is_validation],
            s=36,
            marker="x",
            label="validation",
        )

    finite = np.isfinite(measured) & np.isfinite(predicted)
    if finite.any():
        lo = float(np.nanmin(np.concatenate([measured[finite], predicted[finite]])))
        hi = float(np.nanmax(np.concatenate([measured[finite], predicted[finite]])))
        pad = 0.05 * (hi - lo + 1e-12)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def _scatter_train_val(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    train_label: str = "train / base grid",
    val_label: str = "validation",
) -> None:
    train = df.loc[~df["is_validation"]]
    val = df.loc[df["is_validation"]]
    if not train.empty:
        ax.scatter(
            train[x_col],
            train[y_col],
            s=36,
            alpha=0.85,
            label=train_label,
            zorder=3,
        )
    if not val.empty:
        ax.scatter(
            val[x_col],
            val[y_col],
            s=56,
            marker="x",
            linewidths=1.5,
            label=val_label,
            zorder=4,
        )


def plot_analytical_vs_measured(df: pd.DataFrame, figures_dir: Path = FIGURES_DIR) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    ok = df["oom"] == False
    plot_df = df.loc[ok].copy()
    if plot_df.empty:
        raise RuntimeError("no successful (non-OOM) measurements to plot")

    is_val = plot_df["is_validation"].to_numpy(dtype=bool)

    specs = [
        (
            "latency",
            "real_latency_s",
            "pred_latency_s",
            "Measured latency (s)",
            "Predicted latency (s)",
            "Latency: analytical vs measured",
        ),
        (
            "energy",
            "real_energy_j",
            "pred_energy_j",
            "Measured energy (J)",
            "Predicted energy (J)",
            "Energy: analytical vs measured",
        ),
        (
            "memory",
            "real_memory_bytes",
            "pred_memory_bytes",
            "Measured peak memory (bytes)",
            "Predicted peak memory (bytes)",
            "Memory: analytical vs measured",
        ),
    ]

    if "flops_torch" in plot_df.columns and plot_df["flops_torch"].notna().any():
        specs.append(
            (
                "flops",
                "flops_torch",
                "flops",
                "Torch FlopCounterMode (FLOPs)",
                "Analytical FLOPs",
                "FLOPs: analytical vs torch.utils.flop_counter",
            )
        )

    for name, real_col, pred_col, xlabel, ylabel, title in specs:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        _scatter_with_identity(
            ax,
            plot_df[real_col].to_numpy(dtype=float),
            plot_df[pred_col].to_numpy(dtype=float),
            is_val,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )
        path = figures_dir / f"{name}_pred_vs_measured.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    for size in (32, 224, 512):
        slice_df = plot_df[plot_df["size"] == size].sort_values("batch")
        if slice_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(
            slice_df["batch"],
            slice_df["pred_latency_s"],
            "-",
            color="C0",
            alpha=0.85,
            label="analytical",
        )
        _scatter_train_val(
            ax,
            slice_df,
            "batch",
            "real_latency_s",
            train_label="measured train",
            val_label="measured validation",
        )
        ax.set_xlabel("Batch size B")
        ax.set_ylabel("Latency (s)")
        ax.set_title(f"Latency vs B at S = {size}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = figures_dir / f"latency_vs_B_S{size}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    for size in (32, 224, 512):
        slice_df = plot_df[plot_df["size"] == size].sort_values("batch")
        if slice_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(
            slice_df["batch"],
            slice_df["arithmetic_intensity"],
            "-",
            color="C0",
            alpha=0.7,
            label="analytical curve",
        )
        _scatter_train_val(ax, slice_df, "batch", "arithmetic_intensity")
        ax.set_xlabel("Batch size B")
        ax.set_ylabel("Arithmetic intensity (FLOP/byte)")
        ax.set_title(f"Arithmetic intensity vs B at S = {size}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = figures_dir / f"arithmetic_intensity_vs_B_S{size}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    for batch in (1, 16, 256):
        slice_df = plot_df[plot_df["batch"] == batch].sort_values("size")
        if slice_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(
            slice_df["size"],
            slice_df["arithmetic_intensity"],
            "-",
            color="C0",
            alpha=0.7,
            label="analytical curve",
        )
        _scatter_train_val(ax, slice_df, "size", "arithmetic_intensity")
        ax.set_xlabel("Image size S (px)")
        ax.set_ylabel("Arithmetic intensity (FLOP/byte)")
        ax.set_title(f"Arithmetic intensity vs S at B = {batch}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = figures_dir / f"arithmetic_intensity_vs_S_B{batch}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    return saved


def main() -> dict:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    if not THETA_PATH.exists():
        raise SystemExit(f"missing {THETA_PATH}; run calibrate.py first")

    device = torch.device("cuda")
    model = Model().to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params}")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    latency_theta, energy_theta = load_thetas(THETA_PATH)

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            b = random.choice([1, 2, 4, 8])
            img_size = random.choice([32, 64, 128])
            model(torch.randn(b, 3, img_size, img_size, device=device))
    torch.cuda.synchronize()

    image_sizes, batches, validation = build_grid()
    rows: list[dict] = []

    with torch.inference_mode():
        for image_size in tqdm(image_sizes, desc="image sizes"):
            for batch in batches:
                is_validation = (image_size, batch) in validation
                pred_flops = float(flops(image_size, batch))
                pred_bytes = float(bytes_moved(image_size, batch))
                pred_memory = float(memory(image_size, batch))
                pred_latency = float(latency(image_size, batch, latency_theta))
                pred_energy = float(energy(image_size, batch, energy_theta, latency_theta))
                intensity = float(arithmetic_intensity(image_size, batch))

                try:
                    x = torch.randn(batch, 3, image_size, image_size, device=device)

                    torch_flops = measure_torch_flops(model, x)
                    real_memory = measure_peak_memory_bytes(model, x)
                    real_latency = measure_median_latency_s(model, x, LATENCY_REPEATS)
                    real_energy = measure_energy_j(model, x, handle, ENERGY_REPS)

                    rows.append(
                        {
                            "size": image_size,
                            "batch": batch,
                            "flops": pred_flops,
                            "flops_torch": torch_flops,
                            "bytes_moved": pred_bytes,
                            "arithmetic_intensity": intensity,
                            "pred_memory_bytes": pred_memory,
                            "real_memory_bytes": real_memory,
                            "real_latency_s": real_latency,
                            "pred_latency_s": pred_latency,
                            "real_energy_j": real_energy,
                            "pred_energy_j": pred_energy,
                            "oom": False,
                            "is_validation": is_validation,
                        }
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rows.append(
                        {
                            "size": image_size,
                            "batch": batch,
                            "flops": pred_flops,
                            "flops_torch": None,
                            "bytes_moved": pred_bytes,
                            "arithmetic_intensity": intensity,
                            "pred_memory_bytes": pred_memory,
                            "real_memory_bytes": None,
                            "real_latency_s": None,
                            "pred_latency_s": pred_latency,
                            "real_energy_j": None,
                            "pred_energy_j": pred_energy,
                            "oom": True,
                            "is_validation": is_validation,
                        }
                    )
                    print(f"OOM at S={image_size}, B={batch}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(MEASUREMENTS_PATH, index=False)
    print(f"saved {MEASUREMENTS_PATH} ({len(df)} rows, {int(df['oom'].sum())} OOM)")

    metrics = save_metrics(df)
    print(f"saved {METRICS_PATH}")
    for split in ("all", "train", "validation"):
        lat = metrics[split]["latency_s"]
        print(
            f"  {split}: latency RMSE={lat['rmse']:.6g} s, "
            f"MAPE={lat['mape']:.3%} (n={lat['n']})"
        )

    paths = plot_analytical_vs_measured(df)
    for path in paths:
        print(f"saved {path}")

    return metrics


if __name__ == "__main__":
    main()

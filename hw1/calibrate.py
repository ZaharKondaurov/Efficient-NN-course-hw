import json
import random
from pathlib import Path

import numpy as np
import pynvml
import torch
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression

from equations import bytes_moved, energy, flops, latency, memory
from models import Model

from tqdm.auto import tqdm


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
THETA_PATH = RESULTS_DIR / "theta.json"

WARMUP_ITERS = 5
LATENCY_REPEATS = 30
ENERGY_REPS = 50


def measure_median_latency_s(model: Model, x: torch.Tensor, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(3):
            model(x)
    times_ms = []
    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    return float(np.median(times_ms) / 1000.0)


def measure_energy_j(model: Model, x: torch.Tensor, handle, n_reps: int) -> float:
    with torch.inference_mode():
        for _ in range(3):
            model(x)
    torch.cuda.synchronize()

    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    with torch.inference_mode():
        for _ in range(n_reps):
            model(x)
    torch.cuda.synchronize()
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    return float((e1 - e0) / 1000.0 / n_reps)


def predict_latency_from_features(theta: np.ndarray, f: np.ndarray, b: np.ndarray) -> np.ndarray:
    return theta[0] + np.maximum(theta[1] * f, theta[2] * b)


def fit_latency_theta(f: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:

    def residual(theta: np.ndarray) -> np.ndarray:
        return predict_latency_from_features(theta, f, b) - y

    y_pos = np.maximum(y, 1e-9)
    theta0 = float(np.median(y_pos[y_pos <= np.median(y_pos)]))
    theta1 = float(np.median(y_pos / np.maximum(f, 1.0)))
    theta2 = float(np.median(y_pos / np.maximum(b, 1.0)))
    x0 = np.array([theta0, theta1, theta2], dtype=float)

    result = least_squares(
        residual,
        x0=x0,
        bounds=(0.0, np.inf),
        method="trf",
    )
    return result.x


def main() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda")
    model = Model().to(device).eval()

    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            b = random.choice([1, 2, 4, 8])
            img_size = random.choice([32, 64, 128])
            x = torch.randn(b, 3, img_size, img_size, device=device)
            model(x)
            
    torch.cuda.synchronize()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    flops_results = []
    memory_results = []
    bytes_moved_results = []
    latency_results = []
    energy_results = []
    sizes = []
    batches = []
    oom_configs = []

    with torch.inference_mode():
        for image_size in [32, 64, 128, 224, 256, 384, 512]:
            for batch in tqdm([1, 2, 4, 8, 16, 32, 64, 128, 256]):
                try:
                    x = torch.randn(batch, 3, image_size, image_size, device=device)

                    for _ in range(WARMUP_ITERS):
                        model(x)
                    torch.cuda.synchronize()

                    latency_s = measure_median_latency_s(model, x, LATENCY_REPEATS)
                    energy_j = measure_energy_j(model, x, handle, ENERGY_REPS)

                    latency_results.append(latency_s)
                    energy_results.append(energy_j)
                    flops_results.append(float(flops(image_size, batch)))
                    memory_results.append(float(memory(image_size, batch)))
                    bytes_moved_results.append(float(bytes_moved(image_size, batch)))
                    sizes.append(image_size)
                    batches.append(batch)

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    oom_configs.append({"S": image_size, "B": batch})
                    print(f"OOM at S={image_size}, B={batch}")
                    continue

    sizes_arr = np.asarray(sizes, dtype=float)
    batches_arr = np.asarray(batches, dtype=float)
    flops_results = np.asarray(flops_results, dtype=float)
    bytes_moved_results = np.asarray(bytes_moved_results, dtype=float)
    latency_results = np.asarray(latency_results, dtype=float)
    energy_results = np.asarray(energy_results, dtype=float)

    theta_latency = fit_latency_theta(flops_results, bytes_moved_results, latency_results)
    latency_pred = latency(sizes_arr, batches_arr, theta_latency)

    X_eng = np.column_stack([latency_pred, flops_results, bytes_moved_results])
    model_energy = LinearRegression(fit_intercept=False).fit(X_eng, energy_results)
    theta_energy = model_energy.coef_
    energy_pred = energy(sizes_arr, batches_arr, theta_energy, theta_latency)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "theta_latency": {
            "theta_0": float(theta_latency[0]),
            "theta_flops": float(theta_latency[1]),
            "theta_bytes": float(theta_latency[2]),
            "model": "theta0 + max(theta_flops*FLOPs, theta_bytes*Bytes)",
            "latency_repeats": LATENCY_REPEATS,
            "warmup_iters": WARMUP_ITERS,
        },
        "theta_energy": {
            "theta_static": float(theta_energy[0]),
            "theta_flops": float(theta_energy[1]),
            "theta_bytes": float(theta_energy[2]),
            "model": "theta_static*Latency + theta_flops*FLOPs + theta_bytes*Bytes",
            "energy_reps": ENERGY_REPS,
        },
        "n_samples": len(latency_results),
        "oom": oom_configs,
        "latency_mae_s": float(np.mean(np.abs(latency_pred - latency_results))),
        "energy_mae_j": float(np.mean(np.abs(energy_pred - energy_results))),
    }
    THETA_PATH.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"saved {THETA_PATH}")
    print("theta_latency:", theta_latency.tolist())
    print("theta_energy:", theta_energy.tolist())
    print("oom configs:", len(oom_configs))


if __name__ == "__main__":
    main()

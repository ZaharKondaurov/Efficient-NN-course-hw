import numpy as np
from numpy.typing import ArrayLike, NDArray


def flops(image_size: ArrayLike, batch: ArrayLike) -> NDArray[np.floating]:
    image_size = np.asarray(image_size, dtype=float)
    batch = np.asarray(batch, dtype=float)
    return 17_714 * batch * image_size * image_size + 313_700 * batch


def memory(image_size: ArrayLike, batch: ArrayLike) -> NDArray[np.floating]:
    image_size = np.asarray(image_size, dtype=float)
    batch = np.asarray(batch, dtype=float)
    return 4_161_296 + 400 * batch + 52 * batch * image_size * image_size


def bytes_moved(image_size: ArrayLike, batch: ArrayLike) -> NDArray[np.floating]:
    image_size = np.asarray(image_size, dtype=float)
    batch = np.asarray(batch, dtype=float)
    return 364 * batch * image_size * image_size + 8592 * batch + 4_161_296


def latency(image_size: ArrayLike, batch: ArrayLike, theta: ArrayLike) -> NDArray[np.floating]:
    theta = np.asarray(theta, dtype=float)
    return theta[0] + np.maximum(
        flops(image_size, batch) * theta[1],
        bytes_moved(image_size, batch) * theta[2],
    )


def arithmetic_intensity(image_size: ArrayLike, batch: ArrayLike) -> NDArray[np.floating]:
    """FLOPs per byte moved: FLOPs(S, B) / Bytes(S, B)."""
    return flops(image_size, batch) / bytes_moved(image_size, batch)


def energy(
    image_size: ArrayLike,
    batch: ArrayLike,
    theta_energy: ArrayLike,
    theta: ArrayLike,
) -> NDArray[np.floating]:
    theta_energy = np.asarray(theta_energy, dtype=float)
    return (
        latency(image_size, batch, theta) * theta_energy[0]
        + flops(image_size, batch) * theta_energy[1]
        + bytes_moved(image_size, batch) * theta_energy[2]
    )
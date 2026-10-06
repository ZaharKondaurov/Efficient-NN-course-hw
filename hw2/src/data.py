"""Datasets: CIFAR-10, ImageNet (ImageFolder) and WikiText-2."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

WIKITEXT2_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/{split}.txt"
)


def _loaders(train_set, test_set, batch_size: int, num_workers: int):
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return (
        DataLoader(train_set, shuffle=True, **common),
        DataLoader(test_set, shuffle=False, **common),
    )


def cifar10_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])
    root = str(Path(data_dir))
    train_set = datasets.CIFAR10(root, train=True, transform=train_tf, download=download)
    test_set = datasets.CIFAR10(root, train=False, transform=test_tf, download=download)
    return _loaders(train_set, test_set, batch_size, num_workers)


def imagenet_loaders(
    data_dir: str | Path,
    batch_size: int = 256,
    num_workers: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """Expects ``data_dir/train`` and ``data_dir/val`` in ImageFolder layout."""
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    root = Path(data_dir)
    missing = [str(root / s) for s in ("train", "val") if not (root / s).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"ImageNet not found: {', '.join(missing)}. ImageNet cannot be downloaded "
            "automatically: get ILSVRC2012 from https://image-net.org/download.php and "
            f"extract it to {root}/train/<class>/*.JPEG and {root}/val/<class>/*.JPEG "
            "(or pass --data-dir)."
        )
    train_set = datasets.ImageFolder(str(root / "train"), transform=train_tf)
    test_set = datasets.ImageFolder(str(root / "val"), transform=test_tf)
    return _loaders(train_set, test_set, batch_size, num_workers)


DATASETS = {
    "cifar10": {"loaders": cifar10_loaders, "num_classes": 10, "stem": "cifar"},
    "imagenet": {"loaders": imagenet_loaders, "num_classes": 1000, "stem": "imagenet"},
}


class Corpus:
    """Word-level WikiText-2 with an ``<eos>`` token per line (PyTorch example)."""

    def __init__(self, data_dir: str | Path, download: bool = True) -> None:
        root = Path(data_dir) / "wikitext-2"
        if download:
            root.mkdir(parents=True, exist_ok=True)
            for split in ("train", "valid", "test"):
                path = root / f"{split}.txt"
                if not path.exists():
                    urllib.request.urlretrieve(WIKITEXT2_URL.format(split=split), path)

        self.word2idx: dict[str, int] = {}
        self.train = self._tokenize(root / "train.txt")
        self.valid = self._tokenize(root / "valid.txt")
        self.test = self._tokenize(root / "test.txt")

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def _tokenize(self, path: Path) -> torch.Tensor:
        ids: list[int] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                for word in line.split() + ["<eos>"]:
                    if word not in self.word2idx:
                        self.word2idx[word] = len(self.word2idx)
                    ids.append(self.word2idx[word])
        return torch.tensor(ids, dtype=torch.long)


def batchify(data: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Token stream -> (num_steps, batch_size), dropping the remainder."""
    n = data.size(0) // batch_size
    return data[: n * batch_size].view(batch_size, n).t().contiguous()


def get_batch(source: torch.Tensor, i: int, bptt: int) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = min(bptt, source.size(0) - 1 - i)
    return source[i : i + seq_len], source[i + 1 : i + 1 + seq_len].reshape(-1)

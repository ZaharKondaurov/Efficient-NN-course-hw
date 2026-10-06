"""ResNets of configurable depth (He et al., 2016) for the ZerO experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STANDARD_DEPTHS: dict[int, tuple[str, list[int]]] = {
    18: ("basic", [2, 2, 2, 2]),
    34: ("basic", [3, 4, 6, 3]),
    50: ("bottleneck", [3, 4, 6, 3]),
    101: ("bottleneck", [3, 4, 23, 3]),
    152: ("bottleneck", [3, 8, 36, 3]),
}


class ScalarAffine(nn.Module):
    """Learnable scalar multiplier and bias used in place of BN (Fixup-style)."""

    def __init__(self, num_features: int | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias


def make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(channels)
    if norm == "none":
        return ScalarAffine(channels)
    raise ValueError(f"Unknown norm={norm!r}; use 'bn' or 'none'")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int, norm: str, rezero: bool = False
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = make_norm(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = make_norm(norm, planes)
        self.downsample = _shortcut(in_planes, planes * self.expansion, stride, norm)
        self.last_conv = self.conv2
        # ReZero: x -> x + α F(x), α = 0 at init (Bachlechner et al., 2020).
        self.alpha: nn.Parameter | None = nn.Parameter(torch.zeros(1)) if rezero else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.alpha is not None:
            out = identity + self.alpha * out
        else:
            out = identity + out
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, in_planes: int, planes: int, stride: int, norm: str, rezero: bool = False
    ) -> None:
        super().__init__()
        width = planes * self.expansion
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = make_norm(norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = make_norm(norm, planes)
        self.conv3 = nn.Conv2d(planes, width, 1, bias=False)
        self.bn3 = make_norm(norm, width)
        self.downsample = _shortcut(in_planes, width, stride, norm)
        self.last_conv = self.conv3
        self.alpha: nn.Parameter | None = nn.Parameter(torch.zeros(1)) if rezero else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.alpha is not None:
            out = identity + self.alpha * out
        else:
            out = identity + out
        return F.relu(out)


def _shortcut(in_planes: int, out_planes: int, stride: int, norm: str) -> nn.Module:
    if stride == 1 and in_planes == out_planes:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False),
        make_norm(norm, out_planes),
    )


BLOCKS: dict[str, type[nn.Module]] = {"basic": BasicBlock, "bottleneck": Bottleneck}


class ResNet(nn.Module):
    def __init__(
        self,
        block: str,
        num_blocks: list[int],
        num_classes: int = 10,
        stem: str = "cifar",
        norm: str = "bn",
        rezero: bool = False,
    ) -> None:
        super().__init__()
        block_cls = BLOCKS[block]
        self.norm = norm
        self.rezero = rezero
        self.in_planes = 64

        if stem == "cifar":
            self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        elif stem == "imagenet":
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unknown stem={stem!r}; use 'cifar' or 'imagenet'")
        self.bn1 = make_norm(norm, 64)

        self.layer1 = self._make_layer(block_cls, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_cls, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_cls, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_cls, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block_cls.expansion, num_classes)

    def _make_layer(
        self, block_cls: type[nn.Module], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block_cls(self.in_planes, planes, s, self.norm, self.rezero))
            self.in_planes = planes * block_cls.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(F.adaptive_avg_pool2d(out, 1), 1)
        return self.fc(out)


def resolve_depth(depth: int, block: str | None = None) -> tuple[str, list[int]]:
    if depth in STANDARD_DEPTHS and (block is None or block == STANDARD_DEPTHS[depth][0]):
        return STANDARD_DEPTHS[depth]

    block = block or "basic"
    layers_per_block = 2 if block == "basic" else 3
    per_stage = 4 * layers_per_block
    if (depth - 2) % per_stage != 0 or depth <= 2:
        raise ValueError(
            f"depth={depth} incompatible with {block} blocks: "
            f"need depth = {per_stage}n + 2"
        )
    n = (depth - 2) // per_stage
    return block, [n, n, n, n]


def resnet(
    depth: int = 18,
    num_classes: int = 10,
    stem: str = "cifar",
    norm: str = "bn",
    block: str | None = None,
    rezero: bool = False,
) -> ResNet:
    block_name, num_blocks = resolve_depth(depth, block)
    return ResNet(
        block_name,
        num_blocks,
        num_classes=num_classes,
        stem=stem,
        norm=norm,
        rezero=rezero,
    )


def resnet18(num_classes: int = 10, **kwargs) -> ResNet:
    return resnet(18, num_classes=num_classes, **kwargs)


def resnet50(num_classes: int = 10, **kwargs) -> ResNet:
    return resnet(50, num_classes=num_classes, **kwargs)

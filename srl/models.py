"""Backbones. ImageNet-pretrained torchvision ResNets with a fresh linear head.

Record the exact weights tag: torchvision's IMAGENET1K_V1 and IMAGENET1K_V2
ResNet-50 weights come from different training recipes and can behave
differently under distribution shift. Prior Waterbirds work used the original
(V1) weights, so V1 is the default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tvm

_ARCHS = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights),
}


def build_model(arch: str = "resnet50", pretrained: bool = True,
                weights: str = "IMAGENET1K_V1", num_classes: int = 2) -> nn.Module:
    if arch not in _ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(_ARCHS)}")
    ctor, weight_enum = _ARCHS[arch]
    model = ctor(weights=weight_enum[weights] if pretrained else None)
    # The new head is initialized from the global RNG, which the trainer seeds
    # with the run's training seed, so paired runs share the same head init.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def forward_with_features(model: nn.Module, x: torch.Tensor):
    """Logits and penultimate (pre-fc) features of a torchvision ResNet in
    one pass. Features are what DFR retrains on and what representation
    probes read, so every run stores them at the end of training."""
    m = model
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    feats = torch.flatten(m.avgpool(x), 1)
    return m.fc(feats), feats

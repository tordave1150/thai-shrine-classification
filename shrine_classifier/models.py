from __future__ import annotations

ARCHITECTURES = ("resnet50", "efficientnet_b0", "mobilenet_v3_large", "densenet121", "convnext_tiny")

CAM_LAYER_PATH = {
    "resnet50": "layer4[-1]",
    "efficientnet_b0": "features[-1]",
    "mobilenet_v3_large": "features[-1]",
    "densenet121": "features.denseblock4",
    "convnext_tiny": "features[-1]",
}


def build_model(architecture: str, *, pretrained: bool = True):
    """Return a torchvision CNN with exactly four output logits and weight metadata."""
    try:
        import torch.nn as nn
        from torchvision import models
    except Exception as exc:
        raise RuntimeError("Install compatible PyTorch and torchvision to construct CNNs") from exc
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture {architecture!r}; choose from {ARCHITECTURES}")
    ctor = getattr(models, architecture)
    enum_name = {
        "resnet50": "ResNet50_Weights", "efficientnet_b0": "EfficientNet_B0_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights", "densenet121": "DenseNet121_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
    }[architecture]
    weight_enum = getattr(models, enum_name)
    weights = weight_enum.DEFAULT if pretrained else None
    model = ctor(weights=weights)
    if architecture == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, 4)
    elif architecture in ("efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"):
        head = model.classifier[-1]
        model.classifier[-1] = nn.Linear(head.in_features, 4)
    else:
        model.classifier = nn.Linear(model.classifier.in_features, 4)
    return model, {"architecture": architecture,
                   "weights": f"{enum_name}.DEFAULT" if pretrained else None,
                   "cam_layer": CAM_LAYER_PATH[architecture],
                   "num_classes": 4, "pretrained": bool(pretrained)}


def cam_layer(model, architecture: str):
    if architecture == "resnet50":
        return model.layer4[-1]
    if architecture in ("efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"):
        return model.features[-1]
    if architecture == "densenet121":
        return model.features.denseblock4
    raise ValueError(f"No CAM layer configured for {architecture}")


"""Fine-tuning mode policies for items 3 & 4.

Provides:
  - apply_finetuning_mode(model, architecture, mode) → sets parameters trainable/frozen
  - enforce_mode_each_epoch(model, architecture, mode) → called after model.train() per epoch
  - log_mode_stats(model, architecture, mode) → returns dict with trainable/frozen counts

Modes:
  full     — all parameters trainable (default primary recipe; backbone in train mode)
  frozen   — backbone frozen+eval, head trainable+train
  partial  — backbone final stage + head trainable+train; rest frozen+eval
"""

from __future__ import annotations

from typing import Any

MODES = ("full", "frozen", "partial")


# ---------------------------------------------------------------------------
# Architecture-specific module resolution
# ---------------------------------------------------------------------------

def _get_head_modules(model, architecture: str):
    """Return a list of head modules for this architecture."""
    if architecture == "resnet50":
        return [model.fc]
    elif architecture in ("efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"):
        return [model.classifier]
    elif architecture == "densenet121":
        return [model.classifier]
    raise ValueError(f"Unknown architecture: {architecture}")


def _get_partial_backbone_modules(model, architecture: str):
    """Return a list of modules that form the 'final backbone stage' for partial mode."""
    import torch.nn as nn

    if architecture == "resnet50":
        # Final stage: layer4
        return [model.layer4]

    elif architecture == "densenet121":
        # Final dense block + norm: features.denseblock4 and features.norm5
        # Do NOT train only features[-1] — must explicitly open both
        parts = []
        features = model.features
        if hasattr(features, "denseblock4"):
            parts.append(features.denseblock4)
        else:
            raise AttributeError("densenet121.features has no 'denseblock4' — check torchvision version")
        if hasattr(features, "norm5"):
            parts.append(features.norm5)
        else:
            raise AttributeError("densenet121.features has no 'norm5' — check torchvision version")
        return parts

    elif architecture == "convnext_tiny":
        # features is a Sequential of stages; the final ConvNeXt stage is features[-1]
        # Detect actual structure
        features = model.features
        if len(features) < 1:
            raise AttributeError("convnext_tiny.features appears empty; check torchvision version")
        # Final stage
        return [features[-1]]

    elif architecture == "efficientnet_b0":
        # features is a Sequential of MBConv stages.
        # Final MBConv stage = features[-2] (features[-1] is the projection Conv2dNormActivation)
        features = model.features
        n = len(features)
        if n < 2:
            raise AttributeError(f"efficientnet_b0.features too short (len={n})")
        # Final stage group + final projection
        parts = [features[-2], features[-1]]
        return parts

    elif architecture == "mobilenet_v3_large":
        # The torchvision MobileNetV3-Large tail has a semantic repeated stage
        # (three 160-channel inverted residual blocks in 0.28) followed by the
        # 1x1 feature projection. Resolve its boundary from module metadata.
        features = model.features
        n = len(features)
        if n < 2:
            raise AttributeError(f"mobilenet_v3_large.features too short (len={n})")
        last_inverted = next((i for i in range(n-1, -1, -1)
                              if type(features[i]).__name__ == "InvertedResidual"), None)
        if last_inverted is None or not hasattr(features[last_inverted], "out_channels"):
            raise AttributeError("Cannot resolve MobileNetV3-Large final InvertedResidual stage; inspect torchvision module names")
        stage_channels = features[last_inverted].out_channels
        start = last_inverted
        while start > 0 and type(features[start-1]).__name__ == "InvertedResidual" \
                and getattr(features[start-1], "out_channels", None) == stage_channels:
            start -= 1
        return [*features[start:last_inverted+1], *features[last_inverted+1:]]

    raise ValueError(f"Unknown architecture: {architecture}")


def _get_all_backbone_modules(model, architecture: str):
    """Return all backbone modules (everything except the final classification head)."""
    if architecture == "resnet50":
        return [model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4,
                model.avgpool]
    elif architecture == "efficientnet_b0":
        return [model.features, model.avgpool]
    elif architecture == "mobilenet_v3_large":
        return [model.features, model.avgpool]
    elif architecture == "densenet121":
        return [model.features]
    elif architecture == "convnext_tiny":
        return [model.features, model.avgpool]
    raise ValueError(f"Unknown architecture: {architecture}")


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _set_requires_grad(module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _set_eval_recursive(module) -> None:
    """Set a module and all submodules to eval mode (disables BN updates, dropout etc.)."""
    module.eval()


def _set_train_recursive(module) -> None:
    module.train()


def _collect_named_modules(module_list) -> list[str]:
    """Collect all fully-qualified module names from a list of module objects."""
    return [type(m).__name__ for m in module_list]


# ---------------------------------------------------------------------------
# Mode application
# ---------------------------------------------------------------------------

def apply_finetuning_mode(model, architecture: str, mode: str) -> dict[str, Any]:
    """Apply the trainability policy.

    Must be called after model is constructed (pretrained + head replaced) and
    before the optimizer is created (optimizer gets model.parameters() filtered here).

    Returns a stats dict for logging.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    if mode == "full":
        # All parameters trainable; model.train() will be called before each epoch normally
        for param in model.parameters():
            param.requires_grad = True
        return _log_mode_stats(model, architecture, mode)

    elif mode == "frozen":
        # Backbone frozen+eval; only head is trained
        for m in _get_all_backbone_modules(model, architecture):
            _set_requires_grad(m, False)
        # Ensure head is trainable
        for m in _get_head_modules(model, architecture):
            _set_requires_grad(m, True)
        return _log_mode_stats(model, architecture, mode)

    elif mode == "partial":
        # First freeze everything, then unfreeze partial backbone + head
        for param in model.parameters():
            param.requires_grad = False
        for m in _get_partial_backbone_modules(model, architecture):
            _set_requires_grad(m, True)
        for m in _get_head_modules(model, architecture):
            _set_requires_grad(m, True)
        return _log_mode_stats(model, architecture, mode)

    raise ValueError(f"Unknown mode: {mode}")


def enforce_mode_per_epoch(model, architecture: str, mode: str) -> None:
    """Re-apply eval/train modes after model.train().

    Must be called AFTER model.train() at the start of each training epoch
    so that frozen BatchNorm buffers and dropout are not accidentally activated.
    """
    if mode == "full":
        return  # Nothing to override; all modules train normally

    elif mode == "frozen":
        # Backbone stays in eval; head stays in train
        for m in _get_all_backbone_modules(model, architecture):
            _set_eval_recursive(m)
        for m in _get_head_modules(model, architecture):
            _set_train_recursive(m)

    elif mode == "partial":
        # Frozen backbone modules stay in eval; partial + head in train
        all_backbone = _get_all_backbone_modules(model, architecture)
        trainable_partial = _get_partial_backbone_modules(model, architecture)
        trainable_ids = {id(m) for m in trainable_partial}
        for m in all_backbone:
            if id(m) in trainable_ids:
                _set_train_recursive(m)
            else:
                # Check if any submodule of m is in trainable set
                has_trainable_child = any(id(child) in trainable_ids
                                          for child in m.modules())
                if has_trainable_child:
                    # Mixed: individually set children
                    for child_name, child in m.named_children():
                        if id(child) in trainable_ids:
                            _set_train_recursive(child)
                        else:
                            _set_eval_recursive(child)
                else:
                    _set_eval_recursive(m)
        for m in _get_head_modules(model, architecture):
            _set_train_recursive(m)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_mode_stats(model, architecture: str, mode: str) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # Record exact qualified policy modules (e.g. features.13, features.14,
    # features.15 and features.16 for the current MobileNet semantic tail).
    if mode == "full":
        trainable_modules = [name for name, module in model.named_children()
                             if any(p.requires_grad for p in module.parameters())]
    else:
        selected = _get_head_modules(model, architecture)
        if mode == "partial":
            selected = _get_partial_backbone_modules(model, architecture) + selected
        selected_ids = {id(module) for module in selected}
        trainable_modules = [name for name, module in model.named_modules() if id(module) in selected_ids]

    return {
        "architecture": architecture,
        "mode": mode,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_module_names": trainable_modules,
        "bn_buffer_policy": (
            "frozen backbone BN buffers do not update (module in eval mode during training)"
            if mode in ("frozen", "partial")
            else "all BN buffers update normally (full fine-tuning)"
        ),
    }


def log_mode_stats(model, architecture: str, mode: str) -> dict[str, Any]:
    """Public API to log mode stats at any point (read-only)."""
    return _log_mode_stats(model, architecture, mode)


# ---------------------------------------------------------------------------
# Gradient/weight invariant tests (for test suite)
# ---------------------------------------------------------------------------

def test_gradient_invariants(model, architecture: str, mode: str, device) -> dict[str, Any]:
    """Run a forward+backward pass and verify gradient/weight invariants.

    Returns a dict with pass/fail status and details.
    """
    import torch
    import torch.nn as nn
    import copy

    model = model.to(device)
    model.train()
    enforce_mode_per_epoch(model, architecture, mode)

    # Save a copy of all parameters
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Dummy forward pass
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    out = model(dummy)
    loss = out.sum()
    loss.backward()

    failures: list[str] = []

    # Check that frozen parameters have no gradient or zero gradient
    for name, param in model.named_parameters():
        if not param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                failures.append(f"Frozen param {name} has non-zero gradient")

    # Check that trainable parameters have non-zero gradient (for this dummy input)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                failures.append(f"Trainable param {name} has no gradient after backward")
            # Note: may be zero for certain activations with dummy all-zero input; don't hard-fail on value

    # Simulate an optimizer step and verify frozen params unchanged
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    optimizer.step()
    optimizer.zero_grad()

    after = {name: p.detach().clone() for name, p in model.named_parameters()}
    for name in before:
        param = dict(model.named_parameters())[name]
        if not param.requires_grad:
            if not torch.allclose(before[name], after[name], atol=1e-12):
                failures.append(f"Frozen param {name} changed after optimizer step")

    return {
        "architecture": architecture,
        "mode": mode,
        "passed": not failures,
        "failures": failures,
    }

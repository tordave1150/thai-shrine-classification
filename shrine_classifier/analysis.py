from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from .data import CLASSES


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0)
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASSES))),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(np.mean(p)), "macro_recall": float(np.mean(r)),
            "macro_f1": float(np.mean(f1)),
            "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(CLASSES)},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).tolist(),
            "classification_report": report}


def summarize_runs(rows: list[dict[str, Any]], metrics: tuple[str, ...] = ("test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1")) -> list[dict[str, Any]]:
    summaries = []
    architectures = list(dict.fromkeys(r["architecture"] for r in rows))
    for arch in architectures:
        subset = [r for r in rows if r["architecture"] == arch and r.get("status", "completed") == "completed"]
        result: dict[str, Any] = {"architecture": arch, "completed_runs": len(subset)}
        for metric in metrics:
            values = [float(r[metric]) for r in subset if r.get(metric) is not None]
            result[metric + "_mean"] = float(np.mean(values)) if values else None
            result[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) >= 2 else None
        summaries.append(result)
    return summaries


def paired_holm_tests(rows: list[dict[str, Any]], reference_architecture: str,
                      *, metrics: tuple[str, ...] = ("test_acc", "test_macro_f1"), alpha: float = .05,
                      architecture_order: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    by_arch_seed: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status", "completed") != "completed" or row.get("seed") is None:
            continue
        by_arch_seed.setdefault(row["architecture"], {})[int(row["seed"])] = row
    if reference_architecture not in by_arch_seed:
        raise ValueError(f"Reference architecture {reference_architecture!r} has no completed runs")
    records = []
    for arch, seed_map in by_arch_seed.items():
        if arch == reference_architecture:
            continue
        shared = sorted(set(seed_map) & set(by_arch_seed[reference_architecture]))
        for metric in metrics:
            if not shared or any(metric not in seed_map[s] or metric not in by_arch_seed[reference_architecture][s] for s in shared):
                records.append({"reference": reference_architecture, "competitor": arch, "metric": metric,
                                "paired_n": len(shared), "mean_difference": None, "t_statistic": None,
                                "p_value": None, "status": "incomplete_pairs"})
                continue
            diffs = np.array([float(by_arch_seed[reference_architecture][s][metric]) - float(seed_map[s][metric]) for s in shared])
            if not np.all(np.isfinite(diffs)) or len(diffs) < 2 or np.std(diffs, ddof=1) == 0:
                records.append({"reference": reference_architecture, "competitor": arch, "metric": metric,
                                "paired_n": len(shared), "mean_difference": float(np.mean(diffs)) if np.all(np.isfinite(diffs)) else None,
                                "t_statistic": None, "p_value": None, "status": "undefined_zero_variance_or_nonfinite"})
                continue
            test = ttest_rel([by_arch_seed[reference_architecture][s][metric] for s in shared],
                             [seed_map[s][metric] for s in shared], alternative="two-sided")
            records.append({"reference": reference_architecture, "competitor": arch, "metric": metric,
                            "paired_n": len(shared), "mean_difference": float(np.mean(diffs)),
                            "t_statistic": float(test.statistic), "p_value": float(test.pvalue), "status": "ok"})
    valid = sorted((r for r in records if r["p_value"] is not None), key=lambda r: r["p_value"])
    remaining = len(metrics) * max(0, len(architecture_order or list(by_arch)) - 1)
    running_max = 0.0
    for row in valid:
        running_max = max(running_max, row["p_value"] * remaining)
        row["p_holm"] = min(1.0, running_max)
        row["significant_holm"] = bool(row["p_holm"] < alpha)
        remaining -= 1
    for row in records:
        row.setdefault("p_holm", None)
        row.setdefault("significant_holm", None)
        row["alpha"] = alpha
    return records


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path: Path, values: list[list[int]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(np.asarray(values), cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix")
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)


def gradcam(model, architecture: str, tensor, class_index: int):
    """Compute a normalized spatial Grad-CAM map; caller aligns it with content_bbox."""
    import torch
    from .models import cam_layer
    activations: list[Any] = []
    gradients: list[Any] = []
    layer = cam_layer(model, architecture)
    forward = layer.register_forward_hook(lambda _m, _x, output: activations.append(output))
    try:
        model.zero_grad(set_to_none=True)
        # Frozen modes have no trainable backbone path. Requiring input
        # gradients keeps the CAM target activations differentiable anyway.
        if not tensor.requires_grad:
            tensor = tensor.detach().requires_grad_(True)
        logits = model(tensor)
        score = logits[:, class_index].sum()
        if not score.requires_grad:
            raise RuntimeError("Grad-CAM target has no gradient")
        activations[-1].register_hook(lambda grad: gradients.append(grad))
        score.backward()
        if not gradients:
            raise RuntimeError("CAM layer produced no gradient")
        activation = activations[-1].detach()
        weights = gradients[-1].detach().mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam - cam.amin(dim=(2, 3), keepdim=True)
        cam = cam / cam.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        result = cam[:, 0]
        if not torch.isfinite(result).all():
            raise RuntimeError("Grad-CAM contains non-finite values")
        return result
    finally:
        forward.remove()

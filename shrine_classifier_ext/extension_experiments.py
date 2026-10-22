"""Extension training runners for items 3 and 4.

Provides fit_extension_run() which wraps the primary fit_run() API with:
  - finetuning mode support (full/frozen/partial)
  - augmentation/class-weight ablation
  - per-epoch mode enforcement (frozen BN invariants)
  - run identification that includes condition+mode
  - no changes to the primary fit_run() code

The shared fresh full baseline is a fresh training run with the locked
recipe (not a reuse of historical final runs). It is shared between
ablation and finetune stages via --resume logic.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from shrine_classifier.analysis import save_csv, save_json, classification_metrics
from shrine_classifier.data import CLASSES
from .finetuning_modes import apply_finetuning_mode, enforce_mode_per_epoch, log_mode_stats

EXTENSION_SEEDS = [42, 123, 456, 789, 1024]
CONDITIONS = ("full", "no_augmentation", "no_class_weights", "frozen", "partial")


# ---------------------------------------------------------------------------
# Internal fit function with mode support
# ---------------------------------------------------------------------------

def _fit_extension_run(
    *,
    root: Path,
    split_csv: Path,
    out: Path,
    architecture: str,
    seed: int,
    config: dict[str, Any],
    condition: str,
    mode: str,
    wandb_project: str,
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    device: str = "auto",
    workers: int = 0,
) -> dict[str, Any]:
    """Train one extension run with mode/condition and evaluate test in one pass."""
    import torch
    import torch.nn as nn
    from shrine_classifier.training import (
        seed_everything, load_split, make_loaders,
        _atomic_checkpoint, _save_test_predictions, _save_curves,
    )
    from shrine_classifier.models import build_model
    from shrine_classifier.tracking import RunLogger

    seed_everything(seed)
    rows = load_split(split_csv)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    augmentation = bool(config.get("augmentation", True))
    use_class_weights = bool(config.get("use_class_weights", True))
    batch_size = int(config.get("batch_size", 16))

    loaders, weights, parts = make_loaders(
        root, rows, size=int(config.get("input_size", 224)),
        batch_size=batch_size, seed=seed, workers=workers,
        include_test=True,
        augmentation=augmentation,
        class_weighted=use_class_weights,
    )

    model, model_meta = build_model(architecture, pretrained=True)
    # Apply fine-tuning mode BEFORE creating optimizer
    mode_stats = apply_finetuning_mode(model, architecture, mode)
    model.to(device_obj)

    if use_class_weights:
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device_obj)
    else:
        weight_tensor = torch.ones(len(CLASSES), dtype=torch.float32, device=device_obj)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction="none")

    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("learning_rate", 1e-4))
    decay = float(config.get("weight_decay", 1e-4))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters for mode={mode}")

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    split_sha256 = hashlib.sha256(split_csv.read_bytes()).hexdigest()

    run_id = f"ext_{condition}_{architecture}_seed{seed}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    run_dir = out / "experiments" / condition / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    cfg = {
        **config,
        **model_meta,
        "stage": f"extension_{condition}",
        "condition": condition,
        "mode": mode,
        "seed": seed,
        "architecture": architecture,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "weight_decay": decay,
        "batch_size": batch_size,
        "epochs": int(config.get("epochs", 30)),
        "early_stopping_patience": int(config.get("patience", 7)),
        "scheduler": "ReduceLROnPlateau(mode=max,factor=0.5,patience=3)",
        "device": str(device_obj),
        "class_weights": weights.tolist(),
        "split_csv": str(split_csv.resolve()),
        "split_sha256": split_sha256,
        "augmentation": augmentation,
        "use_class_weights": use_class_weights,
        "augmentation_policy": "mild train-only" if augmentation else "disabled",
        "class_weight_policy": "inverse train frequency" if use_class_weights else "uniform loss",
        "padding_policy": "neutral ImageNet mean; cleaned source",
        "mode_stats": mode_stats,
    }
    save_json(run_dir / "config.json", cfg)

    logger = RunLogger(
        run_dir / "history.csv", mode=wandb_mode, project=wandb_project,
        group="ablation", name=run_id, config=cfg, entity=wandb_entity,
    )

    history: list[dict[str, Any]] = []
    best_score, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    max_epochs = int(config.get("epochs", 30))
    patience = int(config.get("patience", 7))
    checkpoint = run_dir / f"best_{architecture}_seed{seed}.pt"

    try:
        for epoch in range(1, max_epochs + 1):
            # Training phase with mode enforcement
            model.train()
            enforce_mode_per_epoch(model, architecture, mode)
            train_results = _phase_ext(model, loaders["train"], criterion, optimizer, device_obj, training=True)

            # Validation phase
            model.eval()
            val_results = _phase_ext(model, loaders["validation"], criterion, optimizer, device_obj, training=False)

            scheduler.step(val_results["macro_f1"])
            lr_now = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch,
                "train_loss": train_results["loss"], "train_acc": train_results["acc"],
                "train_macro_f1": train_results["macro_f1"],
                "val_loss": val_results["loss"], "val_acc": val_results["acc"],
                "val_macro_f1": val_results["macro_f1"],
                "learning_rate": lr_now,
            }
            history.append(row)
            logger.log(row)

            improvement = val_results["macro_f1"] > best_score + float(config.get("min_delta", 1e-4))
            checkpoint_better = (
                val_results["macro_f1"] > best_score
                or (val_results["macro_f1"] == best_score and val_results["loss"] < best_loss)
            )
            if checkpoint_better:
                best_score = val_results["macro_f1"]
                best_loss = val_results["loss"]
                best_epoch = epoch
                _atomic_checkpoint(checkpoint, {
                    "model_state": model.state_dict(),
                    "architecture": architecture, "seed": seed,
                    "epoch": epoch, "best_val_macro_f1": best_score,
                    "best_val_loss": best_loss, "config": cfg,
                })
            if improvement:
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        if best_epoch == 0:
            raise RuntimeError("No finite validation checkpoint was selected")

        # Load best checkpoint and evaluate test in one pass
        state = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        model.load_state_dict(state["model_state"])
        model.eval()

        test_results = _phase_ext(model, loaders["test"], criterion, optimizer, device_obj, training=False)
        metrics = classification_metrics(test_results["y_true"], test_results["y_pred"])

        best_val_acc = next(r["val_acc"] for r in history if r["epoch"] == best_epoch)
        result: dict[str, Any] = {
            "architecture": architecture, "seed": seed, "condition": condition, "mode": mode,
            "status": "completed",
            "best_epoch": best_epoch, "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_score, "best_val_loss": best_loss,
            "epochs_completed": len(history),
            "checkpoint": str(checkpoint), "run_dir": str(run_dir),
            "test_acc": metrics["accuracy"],
            "test_macro_precision": metrics["macro_precision"],
            "test_macro_recall": metrics["macro_recall"],
            "test_macro_f1": metrics["macro_f1"],
            "per_class": metrics["per_class"],
            "tracking_mode": logger.mode,
            "tracking_error": logger.tracking_error,
            "mode_stats": mode_stats,
        }
        save_json(run_dir / "test_metrics.json", metrics)
        _save_test_predictions(run_dir / "test_predictions.csv", parts["test"], model, loaders["test"], device_obj)

        from shrine_classifier.analysis import save_confusion_matrix
        save_confusion_matrix(run_dir / "confusion_matrix.png", metrics["confusion_matrix"])
        save_json(run_dir / "run_metrics.json", result)
        _save_curves(run_dir / "curves.png", history)

        logger.log({"epoch": len(history), "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc, "best_val_macro_f1": best_score})
        logger.log({"epoch": len(history), "test_acc": metrics["accuracy"],
                    "test_macro_f1": metrics["macro_f1"]})

        return result

    except Exception:
        logger.finish(exit_code=1)
        raise
    finally:
        logger.finish()


def _phase_ext(model, loader, criterion, optimizer, device, *, training: bool) -> dict[str, Any]:
    """Training/eval phase for extension runs."""
    import torch
    from sklearn.metrics import accuracy_score
    from shrine_classifier.analysis import classification_metrics

    loss_sum, weight_sum = 0.0, 0.0
    y_true, y_pred = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ids, _bbox in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            per_item = criterion(logits, labels)
            if training:
                per_item.mean().backward()
                optimizer.step()
            loss_sum += float(per_item.detach().sum().cpu())
            weight_sum += len(labels)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.detach().argmax(1).cpu().tolist())
    return {
        "loss": loss_sum / max(weight_sum, 1),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": classification_metrics(y_true, y_pred)["macro_f1"],
        "y_true": y_true, "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def _find_resumable_extension_run(
    out_dir: Path,
    architecture: str,
    seed: int,
    condition: str,
    mode: str,
    expected_config: dict[str, Any],
    split_sha256: str,
) -> dict[str, Any] | None:
    """Return a completed extension run if protocol/config/split match."""
    required_keys = {"optimizer", "learning_rate", "weight_decay", "batch_size",
                     "epochs", "input_size", "augmentation", "use_class_weights"}
    for config_path in sorted((out_dir / "experiments" / condition).glob("*/config.json")):
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (cfg.get("architecture") != architecture
                or int(cfg.get("seed", -1)) != seed
                or cfg.get("condition") != condition
                or cfg.get("mode") != mode
                or cfg.get("split_sha256") != split_sha256):
            continue
        if any(cfg.get(k) != expected_config.get(k) for k in required_keys):
            continue
        run_dir = config_path.parent
        metrics_p = run_dir / "run_metrics.json"
        test_p = run_dir / "test_metrics.json"
        ckpt_p = run_dir / f"best_{architecture}_seed{seed}.pt"
        if not (metrics_p.is_file() and test_p.is_file() and ckpt_p.is_file()):
            continue
        try:
            result = json.loads(metrics_p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") != "completed":
            continue
        result["run_dir"] = str(run_dir)
        result["checkpoint"] = str(ckpt_p)
        result["resumed"] = True
        return result
    return None


# ---------------------------------------------------------------------------
# High-level stage runners
# ---------------------------------------------------------------------------

def _condition_to_mode(condition: str) -> str:
    """Map condition name to fine-tuning mode."""
    if condition in ("full", "no_augmentation", "no_class_weights"):
        return "full"
    return condition  # "frozen" or "partial"


def run_extension_ablation(
    *,
    out_dir: Path,
    base_out_dir: Path,
    data_root: Path,
    split_csv: Path,
    selection: dict[str, Any],
    params: dict[str, Any],
    seeds: list[int],
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
    device: str,
    workers: int,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Run shared full baseline + ablation conditions for Item 3.

    Conditions: full (shared baseline), no_augmentation, no_class_weights
    Each for all five seeds.
    """
    architecture = selection["reference_architecture"]
    split_sha256 = hashlib.sha256(split_csv.read_bytes()).hexdigest()
    best = params[architecture]
    base_cfg = {
        "optimizer": best["optimizer"],
        "learning_rate": best["learning_rate"],
        "weight_decay": best["weight_decay"],
        "batch_size": best["batch_size"],
        "epochs": 30,
        "patience": 7,
        "input_size": 224,
        "black_threshold": 10,
        "augmentation": True,
        "use_class_weights": True,
    }

    conditions = [
        ("full",              {**base_cfg}),
        ("no_augmentation",  {**base_cfg, "augmentation": False}),
        ("no_class_weights", {**base_cfg, "use_class_weights": False}),
    ]

    registry: list[dict[str, Any]] = _load_or_init_registry(out_dir)
    results_by_condition: dict[str, list[dict[str, Any]]] = {}

    for condition, cfg in conditions:
        mode = "full"
        cond_results: list[dict[str, Any]] = []
        for seed in seeds:
            seed_i = seed
            # Check resume
            existing = None
            if resume:
                existing = _find_resumable_extension_run(
                    out_dir, architecture, seed_i, condition, mode,
                    cfg, split_sha256
                )

            if dry_run:
                if existing:
                    status = "reusable"
                    cond_results.append({"architecture": architecture, "seed": seed_i,
                                         "condition": condition, "mode": mode,
                                         "status": "dry_run_reusable"})
                else:
                    cond_results.append({"architecture": architecture, "seed": seed_i,
                                         "condition": condition, "mode": mode,
                                         "status": "dry_run_new"})
                continue

            if existing:
                cond_results.append(existing)
                registry.append({"architecture": architecture, "seed": seed_i,
                                   "condition": condition, "mode": mode,
                                   "action": "reused", "run_dir": existing.get("run_dir")})
                continue

            try:
                r = _fit_extension_run(
                    root=data_root, split_csv=split_csv, out=out_dir,
                    architecture=architecture, seed=seed_i,
                    config=cfg, condition=condition, mode=mode,
                    wandb_project=wandb_project, wandb_entity=wandb_entity,
                    wandb_mode=wandb_mode, device=device, workers=workers,
                )
            except Exception as exc:
                r = {
                    "architecture": architecture, "seed": seed_i,
                    "condition": condition, "mode": mode,
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                }
            cond_results.append(r)
            registry.append({
                "architecture": architecture, "seed": seed_i,
                "condition": condition, "mode": mode,
                "action": "trained" if r.get("status") == "completed" else "failed",
                "run_dir": r.get("run_dir"), "error": r.get("error"),
            })

        results_by_condition[condition] = cond_results

    _save_registry(out_dir, registry)
    if not dry_run:
        _save_comparisons(out_dir, results_by_condition, seeds, architecture, selection)
    return results_by_condition


def run_extension_finetune(
    *,
    out_dir: Path,
    base_out_dir: Path,
    data_root: Path,
    split_csv: Path,
    selection: dict[str, Any],
    params: dict[str, Any],
    seeds: list[int],
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
    device: str,
    workers: int,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Reuse shared full baseline + run frozen/partial for Item 4."""
    architecture = selection["reference_architecture"]
    split_sha256 = hashlib.sha256(split_csv.read_bytes()).hexdigest()
    best = params[architecture]
    base_cfg = {
        "optimizer": best["optimizer"],
        "learning_rate": best["learning_rate"],
        "weight_decay": best["weight_decay"],
        "batch_size": best["batch_size"],
        "epochs": 30,
        "patience": 7,
        "input_size": 224,
        "black_threshold": 10,
        "augmentation": True,
        "use_class_weights": True,
    }

    conditions = [
        ("full",    "full",    {**base_cfg}),
        ("frozen",  "frozen",  {**base_cfg}),
        ("partial", "partial", {**base_cfg}),
    ]

    registry = _load_or_init_registry(out_dir)
    results_by_condition: dict[str, list[dict[str, Any]]] = {}

    for condition, mode, cfg in conditions:
        cond_results: list[dict[str, Any]] = []
        for seed in seeds:
            existing = None
            if resume:
                existing = _find_resumable_extension_run(
                    out_dir, architecture, seed, condition, mode,
                    cfg, split_sha256
                )

            if dry_run:
                status = "dry_run_reusable" if existing else "dry_run_new"
                cond_results.append({"architecture": architecture, "seed": seed,
                                      "condition": condition, "mode": mode, "status": status})
                continue

            if existing:
                cond_results.append(existing)
                registry.append({"architecture": architecture, "seed": seed,
                                   "condition": condition, "mode": mode,
                                   "action": "reused", "run_dir": existing.get("run_dir")})
                continue

            try:
                r = _fit_extension_run(
                    root=data_root, split_csv=split_csv, out=out_dir,
                    architecture=architecture, seed=seed,
                    config=cfg, condition=condition, mode=mode,
                    wandb_project=wandb_project, wandb_entity=wandb_entity,
                    wandb_mode=wandb_mode, device=device, workers=workers,
                )
            except Exception as exc:
                r = {
                    "architecture": architecture, "seed": seed,
                    "condition": condition, "mode": mode,
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                }
            cond_results.append(r)
            registry.append({
                "architecture": architecture, "seed": seed,
                "condition": condition, "mode": mode,
                "action": "trained" if r.get("status") == "completed" else "failed",
                "run_dir": r.get("run_dir"), "error": r.get("error"),
            })

        results_by_condition[condition] = cond_results

    _save_registry(out_dir, registry)
    if not dry_run:
        _save_comparisons(out_dir, results_by_condition, seeds, architecture, selection)
    return results_by_condition


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_or_init_registry(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "run_registry.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
    return []


def _save_registry(out_dir: Path, registry: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "run_registry.json", registry)


# ---------------------------------------------------------------------------
# Comparison statistics (paired Holm, mean ± SD)
# ---------------------------------------------------------------------------

def _save_comparisons(
    out_dir: Path,
    results_by_condition: dict[str, list[dict[str, Any]]],
    seeds: list[int],
    architecture: str,
    selection: dict[str, Any],
) -> None:
    """Compute and save paired tests, summaries, and delta tables."""
    from scipy.stats import ttest_rel

    comp_dir = out_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Merge all completed conditions from out_dir / "experiments" so all conditions are included
    merged_results: dict[str, list[dict[str, Any]]] = dict(results_by_condition)
    exp_dir = out_dir / "experiments"
    if exp_dir.exists():
        for cond_folder in sorted(exp_dir.iterdir()):
            if cond_folder.is_dir():
                cond_runs = []
                for run_file in sorted(cond_folder.glob("*/run_metrics.json")):
                    try:
                        data = json.loads(run_file.read_text(encoding="utf-8"))
                        if data.get("status") == "completed":
                            cond_runs.append(data)
                    except Exception:
                        pass
                if cond_runs:
                    merged_results[cond_folder.name] = cond_runs

    # Order conditions in canonical sequence
    order = ["full", "no_augmentation", "no_class_weights", "frozen", "partial"]
    results_by_condition = {c: merged_results[c] for c in order if c in merged_results}
    for c, runs in merged_results.items():
        if c not in results_by_condition:
            results_by_condition[c] = runs

    # Build seed-indexed tables: condition → seed → metrics
    seed_metrics: dict[str, dict[int, dict[str, float]]] = {}
    for condition, runs in results_by_condition.items():
        seed_metrics[condition] = {}
        for r in runs:
            if r.get("status") == "completed":
                s = int(r["seed"])
                seed_metrics[condition][s] = {
                    "test_acc": float(r["test_acc"]),
                    "test_macro_f1": float(r["test_macro_f1"]),
                }

    # Summary table: mean ± SD per condition
    summary_rows: list[dict[str, Any]] = []
    for condition, runs in results_by_condition.items():
        completed = [r for r in runs if r.get("status") == "completed"]
        accs = [float(r["test_acc"]) for r in completed]
        f1s = [float(r["test_macro_f1"]) for r in completed]
        summary_rows.append({
            "architecture": architecture, "condition": condition,
            "n_completed": len(completed),
            "test_acc_mean": float(np.mean(accs)) if accs else None,
            "test_acc_sd": float(np.std(accs, ddof=1)) if len(accs) >= 2 else None,
            "test_macro_f1_mean": float(np.mean(f1s)) if f1s else None,
            "test_macro_f1_sd": float(np.std(f1s, ddof=1)) if len(f1s) >= 2 else None,
        })
    save_csv(comp_dir / "condition_summary.csv", summary_rows)

    # Planned contrasts vs "full" baseline — 8 comparisons (4 variants × 2 metrics)
    # Lock family: no_augmentation-full, no_class_weights-full, frozen-full, partial-full
    # × accuracy, macro_f1
    if "full" not in seed_metrics:
        save_json(comp_dir / "statistical_tests.json", {"status": "no_full_baseline"})
        return

    planned_variants = [c for c in ("no_augmentation", "no_class_weights", "frozen", "partial")
                        if c in seed_metrics]
    metrics_list = ["test_acc", "test_macro_f1"]
    family_size = len(planned_variants) * len(metrics_list)

    test_records: list[dict[str, Any]] = []
    for variant in planned_variants:
        for metric in metrics_list:
            full_map = seed_metrics.get("full", {})
            var_map = seed_metrics.get(variant, {})
            shared_seeds = sorted(set(full_map) & set(var_map))
            if len(shared_seeds) < 2:
                test_records.append({
                    "reference": "full", "variant": variant, "metric": metric,
                    "paired_n": len(shared_seeds),
                    "mean_delta": None, "t_statistic": None, "p_value": None,
                    "p_holm": None, "significant_holm": None,
                    "status": "incomplete_pairs",
                    "note": "delta = variant minus full",
                    "alpha": 0.05,
                })
                continue
            full_vals = [full_map[s][metric] for s in shared_seeds]
            var_vals = [var_map[s][metric] for s in shared_seeds]
            diffs = np.array(var_vals) - np.array(full_vals)  # variant − full
            if not np.all(np.isfinite(diffs)) or np.std(diffs, ddof=1) == 0:
                test_records.append({
                    "reference": "full", "variant": variant, "metric": metric,
                    "paired_n": len(shared_seeds),
                    "mean_delta": float(np.mean(diffs)) if np.all(np.isfinite(diffs)) else None,
                    "t_statistic": None, "p_value": None,
                    "p_holm": None, "significant_holm": None,
                    "status": "zero_variance_or_nonfinite",
                    "note": "delta = variant minus full",
                    "alpha": 0.05,
                })
                continue
            result_t = ttest_rel(var_vals, full_vals, alternative="two-sided")
            test_records.append({
                "reference": "full", "variant": variant, "metric": metric,
                "paired_n": len(shared_seeds),
                "mean_delta": float(np.mean(diffs)),
                "t_statistic": float(result_t.statistic),
                "p_value": float(result_t.pvalue),
                "status": "ok",
                "note": "delta = variant minus full",
                "alpha": 0.05,
            })

    # Holm correction over the 8-comparison family
    valid = [r for r in test_records if r.get("p_value") is not None]
    valid.sort(key=lambda r: r["p_value"])
    remaining = family_size
    running_max = 0.0
    for row in valid:
        running_max = max(running_max, row["p_value"] * remaining)
        row["p_holm"] = min(1.0, running_max)
        row["significant_holm"] = bool(row["p_holm"] < 0.05)
        remaining -= 1
    for row in test_records:
        row.setdefault("p_holm", None)
        row.setdefault("significant_holm", None)

    save_csv(comp_dir / "statistical_tests.csv", test_records)
    save_json(comp_dir / "statistical_tests.json", {
        "records": test_records,
        "family_size": family_size,
        "family_description": "4 variants × 2 metrics; separate from primary 8-comparison family",
        "alpha": 0.05,
        "n_seeds": len(seeds),
        "seed_variation_note": (
            "n=5 reflects seed variation on one fixed split. "
            "This is exploratory analysis on a test split already inspected during final evaluation."
        ),
    })

    # Per-class deltas
    delta_rows: list[dict[str, Any]] = []
    for variant in planned_variants:
        var_runs = {int(r["seed"]): r for r in results_by_condition.get(variant, [])
                    if r.get("status") == "completed"}
        full_runs = {int(r["seed"]): r for r in results_by_condition.get("full", [])
                     if r.get("status") == "completed"}
        shared = sorted(set(var_runs) & set(full_runs))
        for cls in CLASSES:
            for metric in ("precision", "recall", "f1"):
                deltas = []
                for s in shared:
                    vr = var_runs[s].get("per_class", {}).get(cls, {})
                    fr = full_runs[s].get("per_class", {}).get(cls, {})
                    if metric in vr and metric in fr:
                        deltas.append(float(vr[metric]) - float(fr[metric]))
                delta_rows.append({
                    "variant": variant, "class": cls, "metric": metric,
                    "n_pairs": len(deltas),
                    "mean_delta": float(np.mean(deltas)) if deltas else None,
                    "sd_delta": float(np.std(deltas, ddof=1)) if len(deltas) >= 2 else None,
                })
    save_csv(comp_dir / "per_class_deltas.csv", delta_rows)

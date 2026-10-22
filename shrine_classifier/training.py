from __future__ import annotations

import json
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .analysis import classification_metrics, save_csv, save_json
from .data import CLASSES, IMAGENET_MEAN, IMAGENET_STD, letterbox, open_rgb
from .models import build_model
from .tracking import RunLogger
from shrine_classifier_ext.finetuning_modes import apply_finetuning_mode, enforce_mode_per_epoch, log_mode_stats


class ImageDataset(Dataset):
    def __init__(self, root: Path, records: list[dict[str, str]], *, training: bool, size: int = 224,
                 augmentation: bool = True):
        self.root, self.records, self.training, self.size = root, records, training, size
        aug = [transforms.RandomRotation(12, fill=(124, 116, 104)),
               transforms.RandomAffine(degrees=0, translate=(.05, .05), scale=(.92, 1.08), fill=(124, 116, 104)),
               transforms.RandomPerspective(distortion_scale=.10, p=.20, fill=(124, 116, 104)),
               transforms.RandomHorizontalFlip(.5), transforms.ColorJitter(.10, .10, .10, .02)]
        base = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if training and augmentation:
            neutral_normalized = [((pixel / 255.0) - mean) / std for pixel, mean, std in
                                  zip((124, 116, 104), IMAGENET_MEAN, IMAGENET_STD)]
            self.transform = transforms.Compose(aug + base + [transforms.RandomErasing(
                p=.05, scale=(.02, .08), value=neutral_normalized)])
        else:
            self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = open_rgb(self.root / row["path"])
        canvas, bbox = letterbox(image, self.size)
        return self.transform(canvas), int(row["label_index"]), row["image_id"], bbox


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def load_split(path: Path) -> list[dict[str, str]]:
    import csv
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical split {path}; run: python pipeline.py --stage prep")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_id", "path", "label", "label_index", "group_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Split must include {sorted(required)}")
    if any(r["label"] not in CLASSES or int(r["label_index"]) != CLASSES.index(r["label"]) for r in rows):
        raise ValueError("Split label/index does not match fixed CLASSES order")
    group_split: dict[str, str] = {}
    for r in rows:
        previous = group_split.setdefault(r["group_id"], r["split"])
        if previous != r["split"]:
            raise ValueError(f"Duplicate group crosses split boundary: {r['group_id']}")
    if {r["split"] for r in rows} != {"train", "validation", "test"}:
        raise ValueError("Canonical split must contain train, validation, and test")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs in canonical split")
    for split in ("train", "validation", "test"):
        labels = {r["label"] for r in rows if r["split"] == split}
        if labels != set(CLASSES):
            raise ValueError(f"All four classes are required in {split}; found {sorted(labels)}")
    return rows


def split_records(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    return {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}


def make_loaders(root: Path, rows: list[dict[str, str]], *, size: int, batch_size: int,
                 seed: int, workers: int, include_test: bool = False,
                 augmentation: bool = True, class_weighted: bool = True):
    parts = split_records(rows)
    gen = torch.Generator().manual_seed(seed)
    loaders = {}
    for split in ("train", "validation") + (("test",) if include_test else ()):
        dataset = ImageDataset(root, parts[split], training=(split == "train"), size=size,
                               augmentation=augmentation)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=workers, pin_memory=torch.cuda.is_available(),
                                    worker_init_fn=worker_init, generator=gen,
                                    persistent_workers=workers > 0, drop_last=False)
    counts = np.bincount([int(r["label_index"]) for r in parts["train"]], minlength=4)
    if (counts == 0).any():
        raise ValueError("Every class needs training examples to compute class weights")
    weights = len(parts["train"]) / (len(CLASSES) * counts.astype(np.float64))
    if not class_weighted:
        weights = np.ones(len(CLASSES), dtype=np.float64)
    return loaders, weights.astype(np.float32), parts


def _phase(model, loader, criterion, optimizer, device, *, training: bool,
           architecture: str | None = None, mode: str | None = None):
    model.train(training)
    if training and mode is not None:
        # The shared phase resets train/eval flags. Reapply the mode here so
        # frozen BatchNorm and dropout modules never drift back into train().
        enforce_mode_per_epoch(model, architecture, mode)
    loss_sum, weight_sum = 0.0, 0.0
    y_true, y_pred, probabilities = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, _ids, _bbox in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
            if not training:
                probabilities.extend(torch.softmax(logits.detach(), dim=1).cpu().tolist())
    return {"loss": loss_sum / max(weight_sum, 1),
            "acc": float(accuracy_score(y_true, y_pred)),
            "macro_f1": classification_metrics(y_true, y_pred)["macro_f1"],
            "y_true": y_true, "y_pred": y_pred, "probabilities": probabilities}


def _atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def fit_run(*, root: Path, split_csv: Path, out: Path, architecture: str, seed: int,
            config: dict[str, Any], stage: str, wandb_project: str,
            wandb_entity: str | None = None,
            wandb_mode: str = "online", device: str = "auto", workers: int = 0,
            evaluate_test: bool = False, active_wandb_run=None,
            mode: str | None = None, run_identity: str | None = None) -> dict[str, Any]:
    seed_everything(seed)
    rows = load_split(split_csv)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device_obj = torch.device(device)
    batch_size = int(config.get("batch_size", 16))
    # Test remains a split manifest row set only until a selected validation
    # checkpoint has been restored below; no test DataLoader exists during fit.
    loaders, weights, parts = make_loaders(root, rows, size=int(config.get("input_size", 224)),
                                           batch_size=batch_size, seed=seed, workers=workers,
                                           include_test=False,
                                           augmentation=bool(config.get("augmentation", True)),
                                           class_weighted=bool(config.get("use_class_weights", True)))
    model, model_meta = build_model(architecture, pretrained=True)
    model.to(device_obj)
    mode_stats = apply_finetuning_mode(model, architecture, mode) if mode else None
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device_obj)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor, reduction="none")
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    lr, decay = float(config.get("learning_rate", 1e-4)), float(config.get("weight_decay", 1e-4))
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError(f"Mode {mode!r} left {architecture} with no trainable parameters")
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(trainable_parameters, lr=lr, momentum=.9, nesterov=True, weight_decay=decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=3)
    identity_suffix = run_identity or f"seed{seed}"
    mode_token = f"_{mode}" if mode else ""
    run_id = f"{stage}_{architecture}{mode_token}_{identity_suffix}_seed{seed}_attempt{int(time.time())}_{uuid.uuid4().hex[:6]}"
    run_dir = out / stage / architecture / mode / run_id if mode else out / stage / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = {**config, **model_meta, "stage": stage, "mode": mode, "mode_stats": mode_stats,
           "run_identity": run_identity, "seed": seed, "optimizer": optimizer_name,
           "learning_rate": lr, "weight_decay": decay, "batch_size": batch_size,
           "epochs": int(config.get("epochs", 30)), "early_stopping_patience": int(config.get("patience", 7)),
           "scheduler": "ReduceLROnPlateau(mode=max,factor=0.5,patience=3)", "device": str(device_obj),
           "class_weights": weights.tolist(), "split_csv": str(split_csv.resolve()),
           "split_sha256": __import__("hashlib").sha256(split_csv.read_bytes()).hexdigest(),
           "black_threshold": int(config.get("black_threshold", 10)),
           "padding_policy": "neutral ImageNet mean; cleaned source",
           "class_weight_policy": "inverse train frequency" if config.get("use_class_weights", True) else "uniform loss",
           "augmentation_policy": "mild train-only" if config.get("augmentation", True) else "disabled"}
    save_json(run_dir / "config.json", cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg["trainable_parameters"] = trainable
    save_json(run_dir / "config.json", cfg)
    group_name = "sweep" if stage in {"tune", "tuning"} else ("ablation" if stage.startswith("ablation") else stage)
    logger = RunLogger(run_dir / "history.csv", mode=wandb_mode, project=wandb_project,
                       group=group_name, name=run_id, config=cfg, entity=wandb_entity,
                       active_run=active_wandb_run)
    history = []
    best_score, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    max_epochs, patience = int(config.get("epochs", 30)), int(config.get("patience", 7))
    checkpoint = run_dir / f"best_{architecture}{mode_token}_seed{seed}.pt"
    try:
        for epoch in range(1, max_epochs + 1):
            train = _phase(model, loaders["train"], criterion, optimizer, device_obj, training=True,
                           architecture=architecture, mode=mode)
            val = _phase(model, loaders["validation"], criterion, optimizer, device_obj, training=False)
            lr_now = float(optimizer.param_groups[0]["lr"])
            scheduler.step(val["macro_f1"])
            row = {"epoch": epoch, "train_loss": train["loss"], "train_acc": train["acc"],
                   "train_macro_f1": train["macro_f1"], "val_loss": val["loss"],
                   "val_acc": val["acc"], "val_macro_f1": val["macro_f1"], "learning_rate": lr_now}
            history.append(row)
            logger.log(row)
            improvement = val["macro_f1"] > best_score + float(config.get("min_delta", 1e-4))
            checkpoint_better = val["macro_f1"] > best_score or (val["macro_f1"] == best_score and val["loss"] < best_loss)
            if checkpoint_better:
                best_score, best_loss, best_epoch = val["macro_f1"], val["loss"], epoch
                _atomic_checkpoint(checkpoint, {"model_state": model.state_dict(), "architecture": architecture,
                                                "mode": mode, "seed": seed, "epoch": epoch, "best_val_macro_f1": best_score,
                                                "best_val_loss": best_loss, "config": cfg})
            if improvement:
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_epoch == 0:
            raise RuntimeError("No finite validation checkpoint was selected")
        state = torch.load(checkpoint, map_location=device_obj, weights_only=False)
        model.load_state_dict(state["model_state"])
        result: dict[str, Any] = {"architecture": architecture, "mode": mode, "seed": seed, "status": "completed",
                                  "trial": config.get("trial"), "run_identity": run_identity,
                                  "best_epoch": best_epoch, "best_val_acc": next(r["val_acc"] for r in history if r["epoch"] == best_epoch),
                                  "best_val_macro_f1": best_score, "best_val_loss": best_loss,
                                  "checkpoint": str(checkpoint), "run_dir": str(run_dir), "epochs_completed": len(history),
                                  "tracking_mode": logger.mode, "tracking_error": logger.tracking_error}
        try:
            result["wandb_run_id"] = logger.run.id if logger.run is not None else None
            result["wandb_run_url"] = logger.run.url if logger.run is not None else None
        except Exception:
            result["wandb_run_id"] = None
            result["wandb_run_url"] = None
        logger.log({"epoch": len(history), "best_epoch": best_epoch,
                    "best_val_acc": result["best_val_acc"], "best_val_macro_f1": best_score})
        if evaluate_test:
            from torch.utils.data import DataLoader
            test_dataset = ImageDataset(root, parts["test"], training=False, size=int(config.get("input_size", 224)),
                                        augmentation=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=workers, pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init,
                generator=torch.Generator().manual_seed(seed), persistent_workers=workers > 0, drop_last=False)
            test = _phase(model, test_loader, criterion, optimizer, device_obj, training=False)
            metrics = classification_metrics(test["y_true"], test["y_pred"])
            result.update({"test_loss": test["loss"], "test_acc": metrics["accuracy"], "test_macro_precision": metrics["macro_precision"],
                           "test_macro_recall": metrics["macro_recall"], "test_macro_f1": metrics["macro_f1"],
                           "per_class": metrics["per_class"]})
            save_json(run_dir / "test_metrics.json", metrics)
            prediction_rows = []
            for record, label, pred, probs in zip(parts["test"], test["y_true"], test["y_pred"], test["probabilities"]):
                prediction_rows.append({"image_id": record["image_id"], "true_label": CLASSES[label],
                                        "predicted_label": CLASSES[pred], "confidence": float(max(probs)),
                                        "architecture": architecture, "mode": mode, "seed": seed,
                                        "run_identity": run_identity,
                                        **{f"prob_{c}": float(probs[i]) for i, c in enumerate(CLASSES)}})
            save_csv(run_dir / "test_predictions.csv", prediction_rows)
            from .analysis import save_confusion_matrix
            save_confusion_matrix(run_dir / "confusion_matrix.png", metrics["confusion_matrix"])
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            matrix=np.asarray(metrics["confusion_matrix"],dtype=np.float64)
            row_totals=matrix.sum(axis=1,keepdims=True)
            normalized=np.divide(matrix,row_totals,out=np.zeros_like(matrix),where=row_totals>0)
            fig,ax=plt.subplots(figsize=(6.5,5.5));image=ax.imshow(normalized,cmap="Blues",vmin=0,vmax=1)
            ax.set_xticks(range(len(CLASSES)),CLASSES,rotation=30,ha="right");ax.set_yticks(range(len(CLASSES)),CLASSES)
            ax.set_xlabel("Predicted");ax.set_ylabel("True");ax.set_title("Row-normalized confusion matrix")
            for i,row_values in enumerate(normalized):
                for j,value in enumerate(row_values):ax.text(j,i,f"{value:.2f}",ha="center",va="center",color="black")
            fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(run_dir/"confusion_matrix_normalized.png",dpi=160);plt.close(fig)
            final_log = {"epoch": len(history), **{k: v for k, v in result.items() if k.startswith("test_")}}
            logger.log(final_log)
            for cls, values in metrics["per_class"].items():
                logger.log({"epoch": len(history), f"precision_{cls}": values["precision"],
                            f"recall_{cls}": values["recall"], f"f1_{cls}": values["f1"],
                            f"support_{cls}": values["support"]})
        logger.finish()
        result["tracking_mode"], result["tracking_error"] = logger.mode, logger.tracking_error
        save_json(run_dir / "run_metrics.json", result)
        _save_curves(run_dir / "curves.png", history)
        save_json(run_dir / "checkpoint_metadata.json", {"checkpoint": checkpoint.name, **result,
                                                           "mode_stats": mode_stats, "class_order": CLASSES,
                                                           "split_sha256": cfg["split_sha256"]})
        return result
    except Exception as exc:
        if run_dir.exists():
            try:
                save_json(run_dir / "failure.json", {"architecture": architecture, "mode": mode,
                    "seed": seed, "run_identity": run_identity, "error": f"{type(exc).__name__}: {exc}",
                    "config": cfg, "finished_utc": datetime.now(timezone.utc).isoformat()})
            except Exception:
                pass
        logger.finish(exit_code=1)
        raise
    finally:
        logger.finish()


def _save_test_predictions(path: Path, records, model, loader, device):
    import csv
    rows = []
    model.eval()
    with torch.no_grad():
        for images, labels, ids, _bbox in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            for image_id, label, p in zip(ids, labels.tolist(), probs):
                pred = int(np.argmax(p))
                rows.append({"image_id": image_id, "true_label": CLASSES[label], "predicted_label": CLASSES[pred],
                             "confidence": float(p[pred]), **{f"prob_{c}": float(p[i]) for i, c in enumerate(CLASSES)}})
    save_csv(path, rows)


def _save_curves(path: Path, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, keys, title in zip(axes, [("train_loss", "val_loss"), ("train_acc", "val_acc"),
                                      ("train_macro_f1", "val_macro_f1")], ("Loss", "Accuracy", "Macro F1")):
        for key in keys:
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)

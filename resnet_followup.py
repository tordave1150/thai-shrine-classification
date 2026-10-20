"""ResNet50/Full follow-up on the completed outputs_modes_v3_td experiment.

Explicit stages only: gradcam, lock, ablation, analyze, report.
No tuning or baseline training. Ablation adds two conditions x five matched
seeds (10 fresh ImageNet runs, at most 300 epochs). The user runs this CLI.
The original selection.json, registry, final results and statistical families
are never rewritten. This study is exploratory after test-informed selection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pipeline_modes import _csv, _fingerprint, _holm, _json, _read_csv
from shrine_classifier.analysis import classification_metrics
from shrine_classifier.data import CLASSES, sha256_file

SEEDS = [42, 123, 456, 789, 1024]
PAIR = {"architecture": "resnet50", "mode": "full"}
VARIANTS = {"no_augmentation": {"augmentation": False},
            "no_class_weights": {"use_class_weights": False}}
CONFIG_KEYS = ("learning_rate", "weight_decay", "optimizer", "batch_size", "epochs",
               "patience", "min_delta", "input_size", "augmentation", "use_class_weights", "optimizer_defaults")
METRICS = ("test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1")
FAMILY = "resnet50_full_ablation_variant_minus_baseline_accuracy_macro_f1_4"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def utc():
    return datetime.now(timezone.utc).isoformat()


def code_hash():
    base = Path(__file__).resolve().parent
    files = [Path(__file__).resolve(), base / "pipeline_modes.py", base / "pipeline.py"]
    for folder in ("shrine_classifier", "shrine_classifier_ext"):
        files.extend(sorted((base / folder).glob("*.py")))
    return _fingerprint([(str(p.relative_to(base)).replace("\\", "/"), sha256_file(p)) for p in files])


def locate(root, saved):
    """Resolve a saved path inside this output root, including a moved checkout."""
    root = root.resolve()
    value = str(saved).replace("\\", "/")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.exists() and candidate.resolve().is_relative_to(root):
        return candidate.resolve()
    # Absolute paths in the registry may refer to the original Windows checkout.
    token = "/" + root.name + "/"
    if token in value:
        candidate = root / value.split(token, 1)[1]
        if candidate.exists() and candidate.resolve().is_relative_to(root):
            return candidate.resolve()
    raise FileNotFoundError(f"Artifact is missing or outside {root}: {saved}")


def rel(root, path):
    return str(Path(path).resolve().relative_to(root.resolve())).replace("\\", "/")


def validate_run(run, root, test_rows, expected_config=None):
    """Validate identities, saved files and metrics before baseline reuse/resume."""
    if run.get("status") != "completed" or any(run.get(k) != v for k, v in PAIR.items()):
        raise ValueError("Expected a completed ResNet50/Full run")
    directory = locate(root, run["run_dir"])
    checkpoint = locate(root, run["checkpoint"])
    if checkpoint.parent != directory:
        raise ValueError("Checkpoint/run-directory identity mismatch")
    cfg = read_json(directory / "config.json")
    metadata = read_json(directory / "checkpoint_metadata.json")
    metrics = read_json(directory / "run_metrics.json")
    if metadata.get("class_order") != CLASSES:
        raise ValueError(f"Class order mismatch: {directory}")
    for item in (cfg, metadata, metrics):
        if any(item.get(k) != v for k, v in PAIR.items()) or int(item.get("seed", -1)) != int(run["seed"]):
            raise ValueError(f"Saved run identity mismatch: {directory}")
    if int(metadata["best_epoch"]) != int(run["best_epoch"]):
        raise ValueError("Selected checkpoint epoch mismatch")
    if metadata.get("split_sha256") != sha256_file(root / "splits.csv"):
        raise ValueError(f"Run split hash mismatch: {directory}")
    if expected_config is not None:
        for key, value in expected_config.items():
            if cfg.get(key) != value:
                raise ValueError(f"Config mismatch for {key}: {directory}")
    predictions = _read_csv(directory / "test_predictions.csv")
    if [r["image_id"] for r in predictions] != [r["image_id"] for r in test_rows]:
        raise ValueError(f"Canonical test ID order mismatch: {directory}")
    for pred, truth in zip(predictions, test_rows):
        if pred["true_label"] != truth["label"] or pred["predicted_label"] not in CLASSES:
            raise ValueError("Prediction label mismatch")
        probs = np.array([float(pred[f"prob_{c}"]) for c in CLASSES])
        if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or not np.isclose(probs.sum(), 1, atol=1e-5):
            raise ValueError("Invalid saved probability vector")
        if CLASSES[int(probs.argmax())] != pred["predicted_label"] or not math.isclose(float(pred["confidence"]), float(probs.max()), abs_tol=1e-6):
            raise ValueError("Saved confidence/prediction mismatch")
    measured = classification_metrics([CLASSES.index(p["true_label"]) for p in predictions],
                                      [CLASSES.index(p["predicted_label"]) for p in predictions])
    for metric in METRICS:
        key = "accuracy" if metric == "test_acc" else metric.removeprefix("test_")
        if not math.isclose(float(run[metric]), measured[key], abs_tol=1e-9):
            raise ValueError(f"Prediction-derived {metric} mismatch: {directory}")
    artifacts = {rel(root, p): sha256_file(p) for p in
                 (checkpoint, directory / "config.json", directory / "checkpoint_metadata.json",
                  directory / "run_metrics.json", directory / "test_metrics.json", directory / "test_predictions.csv")}
    return {**run, "run_dir": str(directory), "checkpoint": str(checkpoint),
            "per_class": measured["per_class"]}, cfg, predictions, artifacts


def context(args):
    root = args.out_dir.resolve()
    from pipeline import _check_existing_split
    _check_existing_split(root, args.data_root.resolve())
    selection = read_json(root / "selection.json")
    protocol = read_json(root / "protocol.json")
    payload = {k: v for k, v in selection.items() if k != "selection_fingerprint"}
    if _fingerprint(payload) != selection["selection_fingerprint"]:
        raise ValueError("Original selection lock fingerprint is corrupt")
    if selection["protocol_fingerprint"] != protocol["protocol_fingerprint"]:
        raise ValueError("Original protocol/selection mismatch")
    if selection["split_file_sha256"] != sha256_file(root / "splits.csv"):
        raise ValueError("Canonical split CSV byte hash mismatch")
    if read_json(root / "split_metadata.json")["split_sha256"] != selection["split_assignment_fingerprint"]:
        raise ValueError("Canonical split assignment fingerprint mismatch")
    split = _read_csv(root / "splits.csv")
    if len(split) != 560 or len({r["image_id"] for r in split}) != 560:
        raise ValueError("Expected 560 unique canonical image IDs")
    # Verify the actual image files; never re-audit into or rewrite old outputs.
    source = read_json(root / "source_manifest.json")
    if _fingerprint([(r["image_id"], r["byte_sha256"]) for r in source["images"]]) != selection["source_fingerprint"]:
        raise ValueError("Source manifest fingerprint differs from the original lock")
    by_id = {r["image_id"]: r for r in split}
    if len(source["images"]) != 560:
        raise ValueError("Incomplete source manifest")
    for image in source["images"]:
        row = by_id[image["image_id"]]
        file = (args.data_root.resolve() / image["path"]).resolve()
        if not file.is_relative_to(args.data_root.resolve()) or sha256_file(file) != image["byte_sha256"]:
            raise ValueError(f"Source image hash/path mismatch: {image['image_id']}")
        if row["label"] != image["label"] or row["path"] != image["path"] or CLASSES[int(row["label_index"])] != row["label"]:
            raise ValueError("Source/split class or path mismatch")
    test_rows = [r for r in split if r["split"] == "test"]
    registry = read_json(root / "run_registry.json")
    winners = read_json(root / "tuning" / "best_params_by_arch_mode.json")["pairs"]["resnet50::full"]
    latest = {}
    for run in registry["final_runs"]:
        if run.get("status") == "completed" and all(run.get(k) == v for k, v in PAIR.items()):
            latest[int(run["seed"])] = run
    if sorted(latest) != sorted(SEEDS) or selection["seeds"] != SEEDS:
        raise ValueError("Expected all five original matched final seeds")
    baseline, configs, artifacts = [], [], {}
    for seed in SEEDS:
        row, cfg, _, files = validate_run(latest[seed], root, test_rows)
        for key in ("learning_rate", "weight_decay", "optimizer", "batch_size"):
            if cfg[key] != winners[key]:
                raise ValueError(f"Baseline is not the independently tuned winner: {key}")
        if cfg.get("selection_fingerprint") != selection["selection_fingerprint"] or not cfg.get("augmentation") or not cfg.get("use_class_weights"):
            raise ValueError("Incompatible baseline selection or factors")
        if (cfg.get("protocol_fingerprint") != protocol["protocol_fingerprint"]
                or cfg.get("split_assignment_fingerprint") != selection["split_assignment_fingerprint"]
                or cfg.get("dataset_fingerprint") != selection["source_fingerprint"]
                or cfg.get("weights") != "ResNet50_Weights.DEFAULT" or not cfg.get("pretrained")):
            raise ValueError("Baseline provenance/pretrained weights mismatch")
        baseline.append(row); configs.append({k: cfg[k] for k in CONFIG_KEYS}); artifacts.update(files)
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("Baseline configs differ across seeds")
    if configs[0]["epochs"] != 30 or configs[0]["input_size"] != 224 or configs[0]["patience"] != 7 or configs[0]["min_delta"] != 1e-4:
        raise ValueError("Baseline training policy differs from the approved follow-up")
    return {"root": root, "selection": selection, "protocol": protocol, "baseline": baseline,
            "config": configs[0], "test_rows": test_rows, "artifacts": artifacts,
            "code_fingerprint": code_hash()}


def study_dir(args):
    if not args.study_name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.study_name):
        raise ValueError("Use a lowercase study name with letters, digits, '_' or '-'")
    return args.out_dir.resolve() / "ablation" / args.study_name


def lock_study(args, ctx):
    directory = study_dir(args)
    payload = {"version": 1, **PAIR, "seeds": SEEDS, "representative_seed": 42,
        "baseline_config": ctx["config"], "variants": VARIANTS,
        "selection_context": "ResNet50/Full selected after viewing test results; exploratory follow-up on an already examined split.",
        "original_statistical_reference": ctx["selection"]["reference_pair"],
        "original_selection_fingerprint": ctx["selection"]["selection_fingerprint"],
        "original_protocol_fingerprint": ctx["selection"]["protocol_fingerprint"],
        "split_sha256": sha256_file(ctx["root"] / "splits.csv"),
        "baseline_artifact_sha256": ctx["artifacts"], "code_fingerprint": ctx["code_fingerprint"],
        "baseline_environment_sha256": sha256_file(ctx["root"] / "environment.json"),
        "baseline_reused_runs": 5, "new_runs": 10, "max_new_epochs": 10 * int(ctx["config"]["epochs"]),
        "statistics": {"family_id": FAMILY, "family_size": 4, "metrics": ["test_acc", "test_macro_f1"],
                       "direction": "variant minus baseline", "alternative": "two-sided", "alpha": .05},
        "baseline_sources": [{"seed": r["seed"], "run_dir": rel(ctx["root"], r["run_dir"]),
                              "checkpoint": rel(ctx["root"], r["checkpoint"])} for r in ctx["baseline"]]}
    fingerprint = _fingerprint(payload)
    target = directory / "ablation_lock.json"
    if target.exists():
        previous = read_json(target)
        if previous.get("study_fingerprint") != fingerprint or _fingerprint({k: v for k, v in previous.items() if k not in ("study_fingerprint", "locked_utc")}) != fingerprint:
            raise ValueError("Study lock differs or is corrupt; use a new --study-name")
        return previous
    result = {**payload, "study_fingerprint": fingerprint, "locked_utc": utc()}
    if not args.dry_run:
        _json(target, result)
    return result


def require_lock(args, ctx):
    if not (study_dir(args) / "ablation_lock.json").is_file():
        raise FileNotFoundError("Run --stage lock before training/analyzing this ablation")
    return lock_study(args, ctx)  # Validate all hashes/configs against the immutable lock.


def ablation(args, ctx):
    lock = require_lock(args, ctx)
    directory = study_dir(args)
    registry_path = directory / "run_registry.json"
    registry = read_json(registry_path) if registry_path.exists() else {"study_fingerprint": lock["study_fingerprint"], "attempts": []}
    if registry["study_fingerprint"] != lock["study_fingerprint"]:
        raise ValueError("Ablation registry belongs to a different lock")
    if args.dry_run:
        return {"new_training_started": False, "new_runs": 10, "baseline_reused": 5,
                "max_new_epochs": lock["max_new_epochs"]}
    import torch
    import torchvision
    from shrine_classifier.training import fit_run
    for variable, suffix in (("WANDB_DIR", "runs"), ("WANDB_CACHE_DIR", "cache"), ("WANDB_DATA_DIR", "data")):
        tracking_dir = directory / "wandb" / suffix
        tracking_dir.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(tracking_dir)
    os.environ.setdefault("WANDB_INIT_TIMEOUT", "45")
    split_copy = directory / "splits.csv"
    if not split_copy.exists():
        shutil.copy2(ctx["root"] / "splits.csv", split_copy)
    if sha256_file(split_copy) != lock["split_sha256"]:
        raise ValueError("Ablation split copy hash mismatch")
    _json(directory / "environment.json", {"created_utc": utc(), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "workers": args.num_workers, "amp": False, "disk_free_bytes": shutil.disk_usage(directory).free,
        "deterministic_algorithms": True, "deterministic_warn_only": True, "cudnn_benchmark": False,
        "requested_tracking_mode": args.wandb_mode, "code_fingerprint": ctx["code_fingerprint"],
        "baseline_environment": read_json(ctx["root"] / "environment.json")})
    print(f"Reusing 5 baseline runs; at most 10 new runs / {lock['max_new_epochs']} epochs; free disk {shutil.disk_usage(directory).free / 2**30:.1f} GiB", flush=True)
    for row in registry["attempts"]:
        if row.get("status") == "running":
            row.update(status="interrupted", error="Previous process ended without completion; restarting as a new attempt")
    for condition, overrides in VARIANTS.items():
        for seed in SEEDS:
            ident = f"{condition}::seed{seed}"
            cfg = {**ctx["config"], **overrides, "selection_fingerprint": ctx["selection"]["selection_fingerprint"],
                "protocol_fingerprint": ctx["protocol"]["protocol_fingerprint"],
                "dataset_fingerprint": ctx["protocol"]["source_fingerprint"],
                "split_assignment_fingerprint": ctx["selection"]["split_assignment_fingerprint"],
                "code_fingerprint": ctx["code_fingerprint"], "amp": False,
                "ablation_condition": condition, "ablation_study_fingerprint": lock["study_fingerprint"]}
            cfg_hash = _fingerprint({"config": cfg, "seed": seed})
            done = next((r for r in reversed(registry["attempts"]) if r["identity"] == ident and r.get("status") == "completed"), None)
            if done:
                if not args.resume:
                    raise ValueError(f"{ident} is complete; use --resume to reuse it")
                if done.get("config_fingerprint") != cfg_hash:
                    raise ValueError(f"Incompatible completed ablation config: {ident}")
                validate_run(done, directory, ctx["test_rows"], cfg)
                for file, digest in done["artifact_sha256"].items():
                    if sha256_file(directory / file) != digest:
                        raise ValueError(f"Completed artifact changed: {file}")
                print(f"[resume] {ident}", flush=True)
                continue
            attempt = 1 + sum(r["identity"] == ident for r in registry["attempts"])
            row = {**PAIR, "identity": ident, "seed": seed, "condition": condition, "attempt": attempt,
                   "config": cfg, "config_fingerprint": cfg_hash, "status": "running", "started_utc": utc()}
            registry["attempts"].append(row)
            _json(registry_path, registry)
            try:
                result = fit_run(root=args.data_root.resolve(), split_csv=ctx["root"] / "splits.csv", out=directory,
                    architecture="resnet50", mode="full", seed=seed, config=cfg,
                    stage=f"ablation_{condition}", run_identity=f"{ident.replace('::', '_')}_attempt{attempt:02d}",
                    wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                    wandb_mode=args.wandb_mode, device=args.device, workers=args.num_workers, evaluate_test=True)
                row.update(result)
                _, _, _, artifacts = validate_run(row, directory, ctx["test_rows"], cfg)
                row["artifact_sha256"] = artifacts
            except KeyboardInterrupt:
                row.update(status="interrupted", error="User/process interruption; no optimizer/RNG resume state")
                row["finished_utc"] = utc(); _json(registry_path, registry)
                raise
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=8))
            row["finished_utc"] = utc()
            _json(registry_path, registry)
            _csv(directory / "run_metrics.csv", registry["attempts"])
            print(f"[ablation] {ident} attempt {attempt}: {row['status']}", flush=True)
    complete = {r["identity"] for r in registry["attempts"] if r.get("status") == "completed"}
    _json(directory / "execution_status.json", {"completed": len(complete), "target": 10, "baseline_reused": 5,
        "next_stage": "analyze" if len(complete) == 10 else "ablation --resume"})
    if len(complete) != 10:
        raise RuntimeError(f"Only {len(complete)}/10 ablation runs completed; inspect registry then use --resume")
    return {"completed": 10, "baseline_reused": 5, "next_stage": "analyze"}


def analyze(args, ctx):
    lock = require_lock(args, ctx)
    directory = study_dir(args)
    registry = read_json(directory / "run_registry.json")
    if registry.get("study_fingerprint") != lock["study_fingerprint"]:
        raise ValueError("Ablation registry/lock mismatch")
    groups = {"baseline": ctx["baseline"]}
    for condition, overrides in VARIANTS.items():
        rows = {}
        for row in registry["attempts"]:
            if row.get("condition") == condition and row.get("status") == "completed":
                rows[int(row["seed"])] = row
        if sorted(rows) != sorted(SEEDS):
            raise ValueError(f"Incomplete ablation: {condition} requires all five seeds")
        group = []
        for seed in SEEDS:
            row = rows[seed]
            if row["config"].get("ablation_study_fingerprint") != lock["study_fingerprint"]:
                raise ValueError("Run study fingerprint mismatch")
            validated, _, _, artifacts = validate_run(row, directory, ctx["test_rows"],
                                                     {**ctx["config"], **overrides})
            if artifacts != row.get("artifact_sha256"):
                raise ValueError("Ablation artifact hashes changed")
            group.append(validated)
        groups[condition] = group
    summary, deltas, tests, per_class = [], [], [], []
    baseline = {int(r["seed"]): r for r in groups["baseline"]}
    for condition, rows in groups.items():
        entry = {**PAIR, "condition": condition, "completed_seeds": 5}
        for metric in METRICS:
            vals = np.array([float(r[metric]) for r in rows])
            entry[metric+"_mean"], entry[metric+"_sample_sd"] = float(vals.mean()), float(vals.std(ddof=1))
        summary.append(entry)
        for cls in CLASSES:
            item = {"condition": condition, "class": cls, "support": rows[0]["per_class"][cls]["support"]}
            for metric in ("precision", "recall", "f1"):
                vals = np.array([r["per_class"][cls][metric] for r in rows])
                item[metric+"_mean"], item[metric+"_sample_sd"] = float(vals.mean()), float(vals.std(ddof=1))
            per_class.append(item)
        if condition == "baseline":
            continue
        for row in rows:
            deltas.append({"condition": condition, "seed": row["seed"],
                **{metric+"_delta_pp": 100 * (float(row[metric]) - float(baseline[int(row["seed"])][metric])) for metric in METRICS}})
        from scipy.stats import ttest_rel
        for metric in ("test_acc", "test_macro_f1"):
            matched = {int(r["seed"]): r for r in rows}
            a = np.array([float(matched[s][metric]) for s in SEEDS])
            b = np.array([float(baseline[s][metric]) for s in SEEDS])
            diff = a - b
            item = {"family_id": FAMILY, "condition": condition, "direction": "variant minus baseline", "metric": metric,
                "seed_ids": ",".join(map(str, SEEDS)), "paired_n": 5, "mean_difference": float(diff.mean()),
                "mean_difference_pp": 100*float(diff.mean()), "t_statistic": None, "p_value": None,
                "status": "undefined_zero_variance"}
            if not np.isfinite(diff).all():
                item["status"] = "undefined_nonfinite"
            elif float(diff.std(ddof=1)) != 0:
                result = ttest_rel(a, b, alternative="two-sided")
                item.update(t_statistic=float(result.statistic), p_value=float(result.pvalue), status="ok")
            tests.append(item)
    _holm(tests, 4)
    _csv(directory / "condition_summary.csv", summary)
    _csv(directory / "per_class_summary.csv", per_class)
    _csv(directory / "matched_deltas.csv", deltas)
    _csv(directory / "statistical_tests.csv", tests)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Baseline", "No augmentation", "No class weights"]
    for ax, metric, title in zip(axes, ("test_acc", "test_macro_f1"), ("Accuracy (%)", "Macro F1 (%)")):
        for x, (condition, rows) in enumerate(groups.items()):
            vals = np.array([float(r[metric])*100 for r in rows])
            ax.scatter(x + np.linspace(-.06, .06, 5), vals, color="#3c9659", marker="D", alpha=.4)
            ax.errorbar(x, vals.mean(), yerr=vals.std(ddof=1), fmt="D", color="#26703d", capsize=7)
        ax.set_xticks(range(3), labels); ax.set_ylabel(title); ax.set_ylim(0, 101); ax.grid(axis="y", alpha=.2)
    fig.suptitle("ResNet50 / Full ablation: mean ± sample SD, 5 matched seeds", y=.99)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(directory / "ablation_summary.png", dpi=180); plt.close(fig)
    result = {"study_fingerprint": lock["study_fingerprint"], "completed_new_runs": 10, "baseline_reused": 5,
        "exploratory": True, "figures": ["ablation_summary.png"], "registry_sha256": sha256_file(directory / "run_registry.json"),
        "artifact_sha256": {p.name: sha256_file(p) for p in [directory / n for n in
            ("condition_summary.csv", "per_class_summary.csv", "matched_deltas.csv", "statistical_tests.csv", "ablation_summary.png")]},
        "note": "Seed variation on a previously examined split, not independent external confirmation. Weighted/unweighted loss values are not directly comparable."}
    _json(directory / "analysis.json", result)
    return result


def make_gradcam(args, ctx):
    if args.dry_run:
        return {"model_inference_started": False, **PAIR, "seed": 42, "checkpoint": ctx["baseline"][0]["checkpoint"]}
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from shrine_classifier.analysis import gradcam
    from shrine_classifier.data import open_rgb, letterbox, IMAGENET_MEAN, IMAGENET_STD
    from shrine_classifier.models import build_model
    from shrine_classifier.training import seed_everything
    from torchvision import transforms
    seed_everything(42)
    run = next(r for r in ctx["baseline"] if int(r["seed"]) == 42)
    checkpoint_digest = sha256_file(Path(run["checkpoint"]))
    predictions = _read_csv(Path(run["run_dir"]) / "test_predictions.csv")
    chosen = []
    for cls in CLASSES:
        subset = [r for r in predictions if r["true_label"] == cls]
        correct = sorted((r for r in subset if r["predicted_label"] == cls), key=lambda r: (-float(r["confidence"]), r["image_id"]))
        wrong = sorted((r for r in subset if r["predicted_label"] != cls), key=lambda r: (-float(r["confidence"]), r["image_id"]))
        if correct:
            chosen.append((correct[0], "highest_confidence_correct"))
        if wrong:
            chosen.append((wrong[0], "highest_confidence_misclassified"))
        elif len(correct) > 1:
            chosen.append((min(correct[1:], key=lambda r: (float(r["confidence"]), r["image_id"])), "lowest_confidence_correct_no_errors_in_class"))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model, _ = build_model("resnet50", pretrained=False)
    state = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    if state.get("architecture") != "resnet50" or state.get("mode") != "full" or int(state.get("seed", -1)) != 42 or int(state.get("epoch", -1)) != int(run["best_epoch"]):
        raise ValueError("Checkpoint identity does not match the saved final run")
    model.load_state_dict(state["model_state"], strict=True); model.to(device).eval()
    tensor_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    directory = ctx["root"] / "gradcam" / "resnet50_full" / "seed42" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(len(chosen), 3, figsize=(14, 3.8*len(chosen)), squeeze=False)
    records = []
    lookup = {r["image_id"]: r for r in ctx["test_rows"]}
    try:
        for i, (pred, reason) in enumerate(chosen):
            row = lookup[pred["image_id"]]
            image = open_rgb(args.data_root / row["path"])
            canvas, bbox = letterbox(image, int(ctx["config"]["input_size"]))
            tensor = tensor_transform(canvas).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = model(tensor).softmax(1)[0].cpu().numpy()
            saved_probs = np.array([float(pred[f"prob_{c}"]) for c in CLASSES])
            abs_err = float(np.abs(probs - saved_probs).max())
            rel_err = float((np.abs(probs - saved_probs) / (np.abs(saved_probs) + 1e-8)).max())
            if probs.argmax() != saved_probs.argmax():
                raise ValueError(
                    f"Checkpoint/preprocessing prediction CHANGED class for {pred['image_id']}: "
                    f"saved={CLASSES[saved_probs.argmax()]}, recomputed={CLASSES[probs.argmax()]}, "
                    f"max_abs_err={abs_err:.2e}, max_rel_err={rel_err:.2e}"
                )
            if abs_err > 1e-3:
                print(f"[gradcam] WARNING: large probability discrepancy for {pred['image_id']}: "
                      f"max_abs_err={abs_err:.2e}, max_rel_err={rel_err:.2e} "
                      f"(batched vs single-image float32 rounding; class unchanged)")
            elif abs_err > 5e-5:
                print(f"[gradcam] NOTE: minor float32 rounding for {pred['image_id']}: "
                      f"max_abs_err={abs_err:.2e} (batched vs single-image inference; class unchanged)")
            target = CLASSES.index(pred["predicted_label"])
            with torch.enable_grad():
                cam = gradcam(model, "resnet50", tensor, target)[0].detach().cpu().numpy()
            x1, y1, x2, y2 = bbox
            crop = cam[y1:y2, x1:x2]
            if not np.isfinite(crop).all() or crop.size == 0 or not np.any(crop > 0):
                raise ValueError(f"CAM is empty/non-finite/all-zero for {pred['image_id']}; no valid figure published")
            heat = np.asarray(Image.fromarray(crop.astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR))
            colored = np.uint8(plt.get_cmap("jet")(np.clip(heat, 0, 1))[..., :3]*255)
            overlay = Image.blend(image, Image.fromarray(colored), .42)
            sample = directory / pred["image_id"].replace("/", "__"); sample.mkdir()
            image.save(sample / "original.png"); Image.fromarray(colored).save(sample / "heatmap.png"); overlay.save(sample / "overlay.png")
            np.save(sample / "cam_original_alignment.npy", heat)
            for ax, picture, title in zip(axes[i], (image, Image.fromarray(colored), overlay), ("Original", "Grad-CAM", "Overlay")):
                ax.imshow(picture); ax.axis("off"); ax.set_title(title)
            axes[i, 0].set_title(f"{row['image_id']}\nTrue: {row['label']}", fontsize=10)
            axes[i, 2].set_title(f"Predicted / explained: {pred['predicted_label']}\nConfidence: {float(pred['confidence'])*100:.2f} · {reason}", fontsize=9)
            records.append({**pred, "selection_reason": reason, "explained_target": pred["predicted_label"],
                "content_bbox": list(bbox), "checkpoint": run["checkpoint"], "checkpoint_sha256": checkpoint_digest,
                "sample_directory": rel(ctx["root"], sample)})
        fig.suptitle("ResNet50 / Full · final seed 42 · Grad-CAM at layer4[-1]", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, .985)); fig.savefig(directory / "gradcam_grid.png", dpi=160)
        _csv(directory / "selected_cases.csv", records)
        result = {**PAIR, "seed": 42, "examples": len(records), "checkpoint_sha256": checkpoint_digest,
            "device": str(device), "torch": torch.__version__, "created_utc": utc(),
            "code_fingerprint": ctx["code_fingerprint"], "selection_context": "Post-hoc report-selected model; Grad-CAM is diagnostic, not causal evidence.",
            "directory": rel(ctx["root"], directory), "artifact_sha256": {p.name: sha256_file(p) for p in (directory / "gradcam_grid.png", directory / "selected_cases.csv")}}
        _json(directory / "manifest.json", result)
        _json(directory.parent / "latest.json", result)
        return result
    except Exception as exc:
        _json(directory / "failure.json", {"error": f"{type(exc).__name__}: {exc}", "completed_cases": len(records)})
        raise
    finally:
        plt.close(fig)


def export_report(args, ctx):
    """Insert only measured, verified follow-up figures/tables into local README."""
    import re
    repo = Path(__file__).resolve().parent
    readme = repo / "README.md"
    text = readme.read_text(encoding="utf-8")
    assets = repo / "report" / "report_assets" / "latest"
    assets.mkdir(parents=True, exist_ok=True)
    updated = []
    source_map = {}

    def replace_block(name, body):
        nonlocal text
        pattern = rf"(<!-- resnet-followup:{name}:start -->).*?(<!-- resnet-followup:{name}:end -->)"
        if not re.search(pattern, text, flags=re.S):
            raise ValueError(f"README follow-up markers missing: {name}")
        text = re.sub(pattern, lambda m: m[1] + "\n\n" + body + "\n\n" + m[2], text, flags=re.S)

    latest = ctx["root"] / "gradcam" / "resnet50_full" / "seed42" / "latest.json"
    if latest.is_file():
        manifest = read_json(latest)
        directory = locate(ctx["root"], manifest["directory"])
        if manifest["checkpoint_sha256"] != sha256_file(Path(next(r for r in ctx["baseline"] if int(r["seed"]) == 42)["checkpoint"])):
            raise ValueError("CAM checkpoint changed")
        for name, digest in manifest["artifact_sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"CAM artifact hash mismatch: {name}")
        shutil.copy2(directory / "gradcam_grid.png", assets / "gradcam_resnet50_full_seed42.png")
        shutil.copy2(directory / "selected_cases.csv", repo / "report" / "resnet_gradcam_cases.csv")
        replace_block("gradcam", "![Grad-CAM ของ ResNet50 Full seed 42](report/report_assets/latest/gradcam_resnet50_full_seed42.png)\n\n"
            f"Grad-CAM ของ ResNet50/Full, final seed 42 จำนวน {manifest['examples']} ภาพ อธิบายคลาสที่โมเดลทำนาย เลือกกรณีถูก/ผิดแยกตาม true class; ถ้าไม่มีภาพผิดใช้กรณีถูกที่ confidence ต่ำ แผนที่จัดแนวกลับสู่ภาพต้นฉบับหลังตัด canvas padding ออก เป็นหลักฐานประกอบการวิเคราะห์ ไม่ได้พิสูจน์เหตุผลเชิงสาเหตุของโมเดล [ข้อมูลรายภาพ](report/resnet_gradcam_cases.csv)\n\n"
            "ภาพ MobileNet ด้านล่างเป็นหลักฐานประกอบของ reference เดิม แยกจากโมเดล ResNet ที่เลือกในรายงาน")
        source_map["gradcam"] = manifest; updated.append("gradcam")
    directory = study_dir(args)
    if (directory / "analysis.json").is_file():
        lock = require_lock(args, ctx)
        analysis = read_json(directory / "analysis.json")
        if analysis["study_fingerprint"] != lock["study_fingerprint"] or analysis["registry_sha256"] != sha256_file(directory / "run_registry.json"):
            raise ValueError("Analysis/lock mismatch")
        for name, digest in analysis["artifact_sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Analysis artifact hash mismatch: {name}")
        for name in ("condition_summary.csv", "matched_deltas.csv", "statistical_tests.csv", "per_class_summary.csv"):
            shutil.copy2(directory / name, repo / "report" / ("resnet_ablation_" + name))
        shutil.copy2(directory / "ablation_summary.png", assets / "resnet_ablation_summary.png")
        rows = _read_csv(directory / "condition_summary.csv")
        lines = ["| Condition | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in rows:
            cells = [row["condition"]] + [f"{float(row[m+'_mean'])*100:.2f} ± {float(row[m+'_sample_sd'])*100:.2f}" for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        replace_block("ablation", "ใช้ ResNet50/Full สูตรเดิมและ split เดิม เปลี่ยนทีละปัจจัย: ปิด augmentation หรือปิด class weights ฝึกใหม่จาก ImageNet variant ละ 5 matched seeds รวม 10 runs และใช้ baseline final เดิม 5 seeds โดยไม่ tune ใหม่ ค่าตารางเป็น mean (%) ± sample SD (จุดเปอร์เซ็นต์)\n\n"
            + "\n".join(lines) + "\n\n![ResNet50 Full ablation](report/report_assets/latest/resnet_ablation_summary.png)\n\n"
            "[ผลต่างจับคู่ราย seed](report/resnet_ablation_matched_deltas.csv) · [สถิติครบ 4 comparisons ใน Holm family แยก](report/resnet_ablation_statistical_tests.csv) · [ผลรายคลาส](report/resnet_ablation_per_class_summary.csv)\n\n"
            "ผลนี้เป็น exploratory follow-up หลังเลือก ResNet จาก test บน split ที่เคยตรวจดูแล้ว สถิติอธิบายความแปรปรวนข้าม seeds ไม่ใช่การยืนยันกับข้อมูลภายนอก และ loss ของเงื่อนไขใช้/ไม่ใช้ class weights ไม่อยู่บนสเกลวัตถุประสงค์เดียวกัน")
        source_map["ablation"] = analysis; updated.append("ablation")
    if not updated:
        raise FileNotFoundError("No completed ResNet follow-up artifacts to insert; run gradcam or ablation + analyze first")
    readme.write_text(text, encoding="utf-8")
    (repo / "report" / "report_source.md").write_text("<!-- Generated mirror of ../README.md; edit the canonical README narrative. -->\n" + text.replace("](report/", "]("), encoding="utf-8")
    _json(repo / "report" / "resnet_followup_sources.json", source_map)
    selected_source = repo / "report" / "selected_model_report_sources.json"
    if selected_source.exists():
        original = read_json(selected_source)
        if "gradcam" in source_map:
            original["gradcam_resnet50_full"] = source_map["gradcam"]
        if "ablation" in source_map:
            original["ablation_resnet50_full"] = source_map["ablation"]
        _json(selected_source, original)
    return {"local_readme_updated": updated, "github_push_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("gradcam", "lock", "ablation", "analyze", "report"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs_modes_v3_td"))
    parser.add_argument("--data-root", type=Path, default=Path("dataset_cleaned"))
    parser.add_argument("--study-name", default="resnet50_full_v1")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project", default="thai-shrine-classifier-td")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Read/validate only; never train or infer")
    args = parser.parse_args()
    ctx = context(args)
    if args.dry_run and args.stage in ("analyze", "report"):
        result = {"dry_run": True, "stage": args.stage, "writes_performed": False}
    else:
        result = {"gradcam": make_gradcam, "lock": lock_study, "ablation": ablation,
                  "analyze": analyze, "report": export_report}[args.stage](args, ctx)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent architecture × fine-tuning-mode experiment runner.

Tuning and lock stages never construct a test DataLoader. All new artifacts
live under outputs_modes_v3 by default; historical outputs are read-only inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from shrine_classifier.data import CLASSES, sha256_file
from shrine_classifier.models import ARCHITECTURES

MODES = ("full", "frozen", "partial")
SEEDS = (42, 123, 456, 789, 1024)
PROTOCOL_VERSION = "architecture-mode-5trial-v2"
SEARCH_SPACE = {
    "learning_rate": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-3},
    "weight_decay": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-2},
    "optimizer": {"values": ["adamw", "adam", "sgd"]},
    "batch_size": {"values": [16, 32]},
}
SEARCH_METHOD = "local seeded random search; trials individually tracked as W&B runs"


def _json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\r\n", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _fingerprint(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_manifest(base: Path, data_root: Path, out: Path) -> dict[str, Any]:
    prop_file = base / "dataset_audit" / "image_properties.csv"
    review_file = base / "dataset_audit" / "near_duplicate_review_decisions.csv"
    metadata_file = data_root / "metadata_cleaned.csv"
    if not prop_file.is_file() or not review_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError("Existing dataset audit, duplicate review, or metadata missing; run the verified audit workflow first")
    props = _read_csv(prop_file)
    if len(props) != 560:
        raise ValueError(f"Expected 560 audited images, found {len(props)}")
    meta_rows = _read_csv(metadata_file)
    metadata: dict[tuple[str, str], str] = {}
    for row in meta_rows:
        key = (row["category"], Path(row["filename"]).stem)
        if key in metadata:
            raise ValueError(f"Ambiguous metadata join for {key}")
        metadata[key] = row.get("source_url", "") or ""
    manifest, problems, domains, domains_by_class = [], [], {}, {name:{} for name in CLASSES}
    split_map = {r["image_id"]: r for r in _read_csv(base / "splits.csv")}
    for row in props:
        rel = Path(row["path"])
        if row["label"] not in CLASSES or int(row["label_index"]) != CLASSES.index(row["label"]):
            problems.append({"path": row["path"], "issue": "class-order mismatch"})
            continue
        img = data_root / rel
        if not img.is_file():
            problems.append({"path": row["path"], "issue": "missing image"})
            continue
        actual = sha256_file(img)
        if actual != row.get("byte_sha256"):
            problems.append({"path": row["path"], "issue": "byte hash changed since audit"})
            continue
        key = (row["label"], rel.stem)
        url = metadata.get(key)
        if url is None:
            problems.append({"path": row["path"], "issue": "metadata join missing"})
            continue
        domain = row.get("source_domain", "")
        domains[domain or "unknown"] = domains.get(domain or "unknown", 0) + 1
        domain_name=domain or "unknown"
        domains_by_class[row["label"]][domain_name]=domains_by_class[row["label"]].get(domain_name,0)+1
        manifest.append({"image_id": row["image_id"], "path": row["path"], "label": row["label"],
                         "label_index": int(row["label_index"]), "width": int(row["width"]),
                         "height": int(row["height"]), "mode": row["mode"], "format": row["format"],
                         "byte_sha256": actual, "pixel_sha256": row["pixel_sha256"],
                         "group_id": split_map.get(row["image_id"], {}).get("group_id"),
                         "source_url": url, "source_domain": domain or None})
    ids = [row["image_id"] for row in manifest]
    if problems or len(set(ids)) != 560 or set(metadata) != {(r["label"], Path(r["path"]).stem) for r in manifest}:
        _csv(out / "dataset_audit" / "source_issues.csv", problems or [{"issue": "duplicate IDs or unused metadata join"}])
        raise ValueError(f"Dataset integrity check failed: {len(problems)} issues; see dataset_audit/source_issues.csv")
    _json(out / "source_manifest.json", {"protocol": PROTOCOL_VERSION, "images": manifest,
          "image_count": len(manifest), "metadata_sha256": sha256_file(metadata_file),
          "audit_properties_sha256": sha256_file(prop_file), "duplicate_review_sha256": sha256_file(review_file),
          "source_domain_counts": domains, "source_domain_counts_by_class":domains_by_class,
          "unknown_url_count": sum(not r["source_url"] for r in manifest)})
    return {"images": manifest, "fingerprint": _fingerprint([(r["image_id"], r["byte_sha256"]) for r in manifest])}


def _current_code_fingerprint() -> str:
    base = Path(__file__).resolve().parent
    entries = []
    for folder in (base / "shrine_classifier", base / "shrine_classifier_ext"):
        for path in sorted(folder.glob("*.py")):
            entries.append((str(path.relative_to(base)), sha256_file(path)))
    for path in (base / "pipeline_modes.py", base / "pipeline.py"):
        if path.exists():
            entries.append((str(path.relative_to(base)), sha256_file(path)))
    return _fingerprint(entries)


def _verify_and_prepare(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    from pipeline import _check_existing_split
    from shrine_classifier.training import load_split

    base = args.base_out_dir.resolve()
    _check_existing_split(base, args.data_root.resolve())
    split_file = base / "splits.csv"
    meta_file = base / "split_metadata.json"
    split_rows = load_split(split_file)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if int(meta.get("split_seed", -1)) != args.split_seed or len(split_rows) != 560 or int(meta.get("num_groups", -1)) != 550:
        raise ValueError("Canonical split does not match the reviewed seed/group/image contract; refusing to regenerate")
    if meta.get("group_overlap") is not False:
        raise ValueError("Canonical split metadata reports group overlap")
    # Copy byte-identical canonical split artifacts into the isolated new output root.
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(split_file, out / "splits.csv")
    shutil.copy2(meta_file, out / "split_metadata.json")
    if sha256_file(split_file) != sha256_file(out / "splits.csv"):
        raise IOError("Split copy byte hash mismatch")
    manifest = _source_manifest(base, args.data_root.resolve(), out)
    protocol = {"version": PROTOCOL_VERSION, "architectures": list(args.architectures), "modes": list(args.modes),
        "class_order": CLASSES, "split_seed": args.split_seed, "split_sha256": sha256_file(split_file),
        "assignment_fingerprint": meta["split_sha256"], "source_fingerprint": manifest["fingerprint"],
        "input_size": args.input_size, "max_epochs": args.epochs, "early_stopping_patience": 7,
        "min_delta": 1e-4, "selection_metric": "validation macro F1",
        "search_space": SEARCH_SPACE, "search_method": SEARCH_METHOD, "tuning_seed": args.seed,
        "trials_per_pair": args.trials_per_pair, "final_seeds": args.seeds,
        "wandb_project": args.wandb_project, "created_utc": datetime.now(timezone.utc).isoformat()}
    protocol["protocol_fingerprint"] = _fingerprint(protocol)
    existing_path = out / "protocol.json"
    if existing_path.exists():
        previous = json.loads(existing_path.read_text(encoding="utf-8"))
        if previous.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise ValueError(f"{out} has a different protocol fingerprint; use a new --out-dir")
    _json(existing_path, protocol)
    env = {"python": sys.version, "platform": platform.platform(), "torch": None, "torchvision": None,
           "numpy": np.__version__, "device_requested": args.device, "wandb_mode_requested": args.wandb_mode,
           "timestamp_utc": datetime.now(timezone.utc).isoformat(), "code_fingerprint": _current_code_fingerprint()}
    import torch
    import torchvision
    env.update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "resolved_device": "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)})
    try:
        import wandb
        env["wandb"] = wandb.__version__
    except Exception:
        env["wandb"] = None
    _json(out / "environment.json", env)
    _json(out / "preflight.json", {"approved_budget": {"tuning_runs": len(args.architectures)*len(args.modes)*args.trials_per_pair,
        "final_runs": len(args.architectures)*len(args.modes)*len(args.seeds),
        "maximum_epochs": len(args.architectures)*len(args.modes)*(args.trials_per_pair+len(args.seeds))*args.epochs},
        "split_counts": meta["counts"], "num_groups": meta["num_groups"], "test_accessed": False,
        "source_fingerprint": manifest["fingerprint"], "protocol_fingerprint": protocol["protocol_fingerprint"],
        "dataset_padding_available": args.padded_root.exists() if args.padded_root else False,
        "disk_free_bytes": shutil.disk_usage(out).free, "disk_free_gib": round(shutil.disk_usage(out).free/(1024**3),2)})
    old_audit = base / "dataset_audit"
    if old_audit.is_dir() and not (out / "dataset_audit").exists():
        shutil.copytree(old_audit, out / "dataset_audit")
    # Reuse the existing qualitative ImageNet baseline only as a historical, compatible asset.
    old_baseline = base / "baseline"
    if old_baseline.exists() and not (out / "baseline").exists():
        shutil.copytree(old_baseline, out / "baseline")
    _json(out / "execution_status.json", {"stage": "prepare", "status": "completed",
        "updated_utc": datetime.now(timezone.utc).isoformat(), "completed_tuning_trials": 0,
        "target_tuning_trials": len(args.architectures)*len(args.modes)*args.trials_per_pair,
        "completed_final_runs": 0, "target_final_runs": len(args.architectures)*len(args.modes)*len(args.seeds),
        "split_verified": True, "split_modified": False, "test_predictions_created": False,
        "protocol_fingerprint": protocol["protocol_fingerprint"]})
    return {"protocol": protocol, "metadata": meta, "manifest": manifest, "environment": env}


def _sample_configs(count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    return [{"learning_rate": float(10 ** rng.uniform(-5, -3)),
             "weight_decay": float(10 ** rng.uniform(-6, -2)),
             "optimizer": str(rng.choice(["adamw", "adam", "sgd"])),
             "batch_size": int(rng.choice([16, 32]))} for _ in range(count)]


def _load_registry(out: Path) -> dict[str, Any]:
    path = out / "run_registry.json"
    if path.exists():
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("protocol_fingerprint") != json.loads((out / "protocol.json").read_text(encoding="utf-8"))["protocol_fingerprint"]:
            raise ValueError("Run registry protocol fingerprint mismatch")
        changed = False
        for stage in ("trials", "final_runs"):
            for row in obj.get(stage, []):
                if row.get("status") == "running":
                    row.update({"status": "interrupted", "finished_utc": datetime.now(timezone.utc).isoformat(),
                                "error": row.get("error") or "Process ended before the run recorded a terminal status; recovered on resume."})
                    changed = True
        if changed:
            _json(path, obj)
        return obj
    return {"protocol_fingerprint": json.loads((out / "protocol.json").read_text(encoding="utf-8"))["protocol_fingerprint"], "trials": [], "final_runs": []}


def _write_registry(out: Path, registry: dict[str, Any]) -> None:
    _json(out / "run_registry.json", registry)
    done_tune = len({r.get("identity") for r in registry["trials"] if r.get("status") == "completed"})
    done_final = len({r.get("identity") for r in registry["final_runs"] if r.get("status") == "completed"})
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    tune_target=len(protocol["architectures"])*len(protocol["modes"])*int(protocol["trials_per_pair"])
    final_target=len(protocol["architectures"])*len(protocol["modes"])*len(protocol["final_seeds"])
    _json(out / "execution_status.json", {"updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed_tuning_trials": done_tune, "target_tuning_trials": tune_target,
        "completed_final_runs": done_final, "target_final_runs": final_target,
        "failed_trials": sum(r.get("status") == "failed" for r in registry["trials"]),
        "failed_final_runs": sum(r.get("status") == "failed" for r in registry["final_runs"]),
        "protocol_fingerprint": registry["protocol_fingerprint"]})


def _pair_key(architecture: str, mode: str) -> str:
    return f"{architecture}::{mode}"


def _latest_completed_seed_rows(records: list[dict[str,Any]]) -> list[dict[str,Any]]:
    latest={}
    for row in records:
        if row.get("status")=="completed" and row.get("seed") is not None:
            latest[(row["architecture"],row["mode"],int(row["seed"]))]=row
    return list(latest.values())


def _tune(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    from shrine_classifier.training import fit_run
    from shrine_classifier.analysis import save_csv

    registry = _load_registry(out)
    trial_rows = registry["trials"]
    fixed_configs = _sample_configs(args.trials_per_pair, args.seed)
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    code_hash = _current_code_fingerprint()
    target = len(args.architectures) * len(args.modes) * args.trials_per_pair
    _json(out / "tuning" / "search_definition.json", {"method": SEARCH_METHOD, "search_space": SEARCH_SPACE,
        "seed": args.seed, "trials_per_pair": args.trials_per_pair, "sampled_configs": fixed_configs,
        "unique_trial_target": target, "protocol_fingerprint": protocol["protocol_fingerprint"]})
    for architecture in args.architectures:
        for mode in args.modes:
            for trial_no, sampled in enumerate(fixed_configs, 1):
                ident = f"{architecture}::{mode}::trial{trial_no:03d}"
                cfg = {**sampled, "architecture": architecture, "mode": mode, "seed": args.seed,
                       "trial": trial_no, "epochs": args.epochs, "patience": 7, "min_delta": 1e-4,
                       "input_size": args.input_size, "augmentation": True, "use_class_weights": True,
                       "optimizer_defaults": {"sgd_momentum": .9, "sgd_nesterov": True,
                                              "adam_betas": [0.9, .999], "epsilon": 1e-8,
                                              "weight_decay_mode": "optimizer default"}}
                cfg.update({"protocol_fingerprint":protocol["protocol_fingerprint"],"dataset_fingerprint":protocol["source_fingerprint"],
                            "split_assignment_fingerprint":protocol["assignment_fingerprint"],"code_fingerprint":code_hash,"amp":False})
                cfg_hash = _fingerprint({"cfg": cfg, "split": protocol["assignment_fingerprint"],
                    "source": protocol["source_fingerprint"], "code": code_hash, "protocol": protocol["protocol_fingerprint"]})
                previous = next((r for r in reversed(trial_rows) if r["identity"] == ident and r.get("config_fingerprint") == cfg_hash and r.get("status") == "completed"), None)
                # Verify completion artifacts without ever borrowing legacy runs.
                if args.resume and previous:
                    rdir = Path(previous.get("run_dir", ""))
                    if (rdir / "run_metrics.json").is_file() and (rdir / "history.csv").is_file() and (rdir / "checkpoint_metadata.json").is_file():
                        print(f"[resume] {ident} already complete", flush=True)
                        continue
                attempt=1+sum(r.get("identity")==ident for r in trial_rows)
                record = {"identity": ident, "attempt":attempt, "architecture": architecture, "mode": mode,
                          "trial": trial_no, "seed": args.seed, "config": cfg,
                          "config_fingerprint": cfg_hash, "code_fingerprint": code_hash,
                          "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
                trial_rows.append(record)
                _write_registry(out, registry)
                try:
                    result = fit_run(root=args.data_root.resolve(), split_csv=out / "splits.csv", out=out,
                        architecture=architecture, mode=mode, run_identity=f"trial{trial_no:03d}_attempt{attempt:02d}", seed=args.seed,
                        config=cfg, stage="tuning", wandb_project=args.wandb_project,
                        wandb_entity=args.wandb_entity, wandb_mode=args.wandb_mode,
                        device=args.device, workers=args.num_workers, evaluate_test=False)
                    record.update(result)
                    record["status"] = "completed"
                except Exception as exc:
                    record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc(limit=5)})
                record["finished_utc"] = datetime.now(timezone.utc).isoformat()
                _write_registry(out, registry)
                save_csv(out / "tuning" / "all_trials.csv", trial_rows)
                print(f"[tune] {ident}: {record['status']} val_macro_f1={record.get('best_val_macro_f1')} ({sum(x.get('status')=='completed' for x in trial_rows)}/{target} successful)", flush=True)
            successes = [r for r in trial_rows if r["architecture"] == architecture and r["mode"] == mode
                         and r.get("status") == "completed" and r.get("best_val_macro_f1") is not None]
            if not successes:
                _json(out / "tuning" / "all_trials.json", trial_rows)
                raise RuntimeError(f"No successful trial for {architecture}/{mode}; see run_registry.json")
    save_csv(out / "tuning" / "all_trials.csv", trial_rows)
    _json(out / "tuning" / "all_trials.json", trial_rows)
    return {"completed": sum(r.get("status") == "completed" for r in trial_rows), "target": target}


def _lock(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    registry = _load_registry(out)
    rows = registry["trials"]
    winners, selection_rows = {}, []
    for architecture in args.architectures:
        for mode in args.modes:
            sampled_trials={int(r["trial"]) for r in rows if r.get("architecture")==architecture and r.get("mode")==mode}
            expected_trials=set(range(1,args.trials_per_pair+1))
            if sampled_trials!=expected_trials:
                raise RuntimeError(f"Cannot lock {architecture}/{mode}: recorded trial IDs {sorted(sampled_trials)}, expected {sorted(expected_trials)}")
            trials = [r for r in rows if r.get("architecture") == architecture and r.get("mode") == mode
                      and r.get("status") == "completed" and r.get("best_val_macro_f1") is not None]
            if not trials:
                raise RuntimeError(f"Cannot lock: no successful validation winner for {architecture}/{mode}")
            best = sorted(trials, key=lambda r: (-float(r["best_val_macro_f1"]), -float(r["best_val_acc"]),
                float(r["best_val_loss"]), int(r["trial"])))[0]
            # Tuning stores the training recipe inside record['config']; metrics
            # returned by fit_run are merged at the record's top level.
            winning_config = best.get("config")
            recipe_keys = ("learning_rate", "weight_decay", "optimizer", "batch_size")
            if not isinstance(winning_config, dict) or any(k not in winning_config for k in recipe_keys):
                raise ValueError(f"Winning trial {best['identity']} has an incomplete saved config; refusing to infer parameters")
            recipe = {k: winning_config[k] for k in recipe_keys}
            winners[_pair_key(architecture, mode)] = {"architecture": architecture, "mode": mode,
                **recipe, "winning_trial": best["trial"], "best_val_macro_f1": best["best_val_macro_f1"],
                "best_val_acc": best["best_val_acc"], "best_val_loss": best["best_val_loss"],
                "trial_config_fingerprint": best["config_fingerprint"], "tuning_checkpoint": best["checkpoint"]}
            selection_rows.append(winners[_pair_key(architecture, mode)])
    order_arch = {v: i for i, v in enumerate(args.architectures)}
    order_mode = {v: i for i, v in enumerate(args.modes)}
    reference = sorted(selection_rows, key=lambda r: (-float(r["best_val_macro_f1"]),
        -float(r["best_val_acc"]), float(r["best_val_loss"]), order_arch[r["architecture"]], order_mode[r["mode"]]))[0]
    tune_path = out / "tuning" / "best_params_by_arch_mode.json"
    selection_path=out/"selection.json"
    if selection_path.is_file():
        prior=json.loads(selection_path.read_text(encoding="utf-8"))
        prior_pairs=json.loads(tune_path.read_text(encoding="utf-8")).get("pairs",{}) if tune_path.is_file() else {}
        stable=(prior.get("protocol_fingerprint")==protocol["protocol_fingerprint"] and prior_pairs==winners and
                prior.get("reference_pair")=={k:reference[k] for k in ("architecture","mode")})
        if not stable:raise ValueError("A previous selection lock differs; do not overwrite post-lock decisions. Start an explicitly versioned output.")
        return prior
    if tune_path.exists():
        old = json.loads(tune_path.read_text(encoding="utf-8"))
        if old.get("protocol_fingerprint") not in (None, protocol["protocol_fingerprint"]):
            raise ValueError("Tuning winners belong to a different protocol")
    _json(tune_path, {"protocol_fingerprint": protocol["protocol_fingerprint"], "search_space": SEARCH_SPACE,
        "trials_per_pair": args.trials_per_pair, "pairs": winners})
    locked = {"protocol_fingerprint": protocol["protocol_fingerprint"],
        "source_fingerprint": protocol["source_fingerprint"], "split_file_sha256": sha256_file(out / "splits.csv"),
        "split_assignment_fingerprint": protocol["assignment_fingerprint"], "architectures": args.architectures,
        "modes": args.modes, "seeds": args.seeds, "representative_seed": 42 if 42 in args.seeds else args.seeds[0],
        "epochs": args.epochs, "trials_per_pair": args.trials_per_pair,
        "tuned_pair_count": len(winners), "reference_pair": {k: reference[k] for k in ("architecture", "mode")},
        "reference_validation": {k: reference[k] for k in ("best_val_macro_f1", "best_val_acc", "best_val_loss")},
        "primary_holm_family": {"family_id": "reference_vs_other_pairs_accuracy_macro_f1", "planned_size": (len(winners)-1)*2},
        "secondary_holm_family": {"family_id": "within_architecture_mode_pairs_accuracy_macro_f1", "planned_size": len(args.architectures)*len(args.modes)*(len(args.modes)-1)},
        "test_metrics_read": False, "locked_utc": datetime.now(timezone.utc).isoformat()}
    if len(winners) != len(args.architectures) * len(args.modes):
        raise ValueError("Cannot lock incomplete architecture-mode grid")
    locked["selection_fingerprint"] = _fingerprint(locked)
    _json(out / "selection.json", locked)
    return locked


def _verify_lock(args: argparse.Namespace, out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path, winner_path = out / "selection.json", out / "tuning" / "best_params_by_arch_mode.json"
    if not selection_path.is_file() or not winner_path.is_file():
        raise FileNotFoundError("Locked validation winners missing; run --stage lock")
    selection, winners = json.loads(selection_path.read_text(encoding="utf-8")), json.loads(winner_path.read_text(encoding="utf-8"))
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if selection.get("protocol_fingerprint") != protocol["protocol_fingerprint"] or \
       winners.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
        raise ValueError("Selection/winner protocol fingerprint mismatch")
    saved_fingerprint=selection.get("selection_fingerprint")
    fingerprint_payload={k:v for k,v in selection.items() if k!="selection_fingerprint"}
    if saved_fingerprint!=_fingerprint(fingerprint_payload):
        raise ValueError("Selection lock fingerprint is corrupt")
    if selection.get("test_metrics_read") is not False or selection.get("split_file_sha256") != sha256_file(out / "splits.csv"):
        raise ValueError("Selection lock is incompatible, touched-test, or has a changed split")
    expected = len(args.architectures) * len(args.modes)
    if selection.get("tuned_pair_count") != expected or len(winners.get("pairs", {})) != expected:
        raise ValueError("Selection lock does not cover every configured pair")
    if selection["architectures"] != args.architectures or selection["modes"] != args.modes or selection["seeds"] != args.seeds:
        raise ValueError("Final run grid/seeds differ from frozen selection")
    return selection, winners["pairs"]


def _final(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    from shrine_classifier.training import fit_run
    from shrine_classifier.analysis import save_csv

    selection, winners = _verify_lock(args, out)
    registry = _load_registry(out)
    final_rows = registry["final_runs"]
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    code_hash = _current_code_fingerprint()
    target = len(args.architectures) * len(args.modes) * len(args.seeds)
    for architecture in args.architectures:
        for mode in args.modes:
            pair = winners[_pair_key(architecture, mode)]
            for seed in args.seeds:
                ident = f"{architecture}::{mode}::seed{seed}"
                cfg = {k: pair[k] for k in ("learning_rate", "weight_decay", "optimizer", "batch_size")}
                cfg.update({"epochs": args.epochs, "patience": 7, "min_delta": 1e-4, "input_size": args.input_size,
                    "augmentation": True, "use_class_weights": True,
                    "optimizer_defaults": {"sgd_momentum": .9, "sgd_nesterov": True,
                        "adam_betas": [.9,.999], "epsilon": 1e-8}, "selection_fingerprint": selection["selection_fingerprint"]})
                cfg.update({"protocol_fingerprint":protocol["protocol_fingerprint"],"dataset_fingerprint":protocol["source_fingerprint"],
                            "split_assignment_fingerprint":protocol["assignment_fingerprint"],"code_fingerprint":code_hash,"amp":False})
                cfg_hash = _fingerprint({"cfg": cfg, "pair_trial": pair["trial_config_fingerprint"],
                    "seed": seed, "split": protocol["assignment_fingerprint"], "source": protocol["source_fingerprint"],
                    "code": code_hash, "protocol": protocol["protocol_fingerprint"]})
                previous = next((r for r in reversed(final_rows) if r["identity"] == ident and r.get("config_fingerprint") == cfg_hash and r.get("status") == "completed"), None)
                if args.resume and previous:
                    run_dir = Path(previous.get("run_dir", ""))
                    if (run_dir / "test_predictions.csv").is_file() and (run_dir / "test_metrics.json").is_file() and (run_dir / "checkpoint_metadata.json").is_file():
                        print(f"[resume] {ident} already complete", flush=True)
                        continue
                attempt=1+sum(r.get("identity")==ident for r in final_rows)
                record = {"identity": ident, "attempt":attempt, "architecture": architecture, "mode": mode, "seed": seed,
                    "trial": pair["winning_trial"], "config": cfg, "config_fingerprint": cfg_hash,
                    "code_fingerprint": code_hash, "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
                final_rows.append(record)
                _write_registry(out, registry)
                try:
                    result = fit_run(root=args.data_root.resolve(), split_csv=out / "splits.csv", out=out,
                        architecture=architecture, mode=mode, run_identity=f"final_seed{seed}_attempt{attempt:02d}", seed=seed, config=cfg,
                        stage="final", wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
                        wandb_mode=args.wandb_mode, device=args.device, workers=args.num_workers, evaluate_test=True)
                    record.update(result)
                    record["status"] = "completed"
                except Exception as exc:
                    record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                   "traceback": traceback.format_exc(limit=5)})
                record["finished_utc"] = datetime.now(timezone.utc).isoformat()
                _write_registry(out, registry)
                save_csv(out / "final" / "run_metrics.csv", final_rows)
                print(f"[final] {ident}: {record['status']} test_macro_f1={record.get('test_macro_f1')} ({sum(x.get('status')=='completed' for x in final_rows)}/{target} successful)", flush=True)
    save_csv(out / "final" / "run_metrics.csv", final_rows)
    _summarize(args, out, final_rows, selection)
    return {"completed": sum(r.get("status") == "completed" for r in final_rows), "target": target}


def _holm(rows: list[dict[str, Any]], family_size: int) -> None:
    valid = [(i, float(r["p_value"])) for i, r in enumerate(rows) if r.get("p_value") is not None and np.isfinite(float(r["p_value"]))]
    valid.sort(key=lambda v: v[1])
    running = 0.0
    for rank, (index, p_value) in enumerate(valid):
        running = max(running, p_value * (family_size - rank))
        rows[index]["p_holm"] = min(1., running)
        rows[index]["significant_holm"] = bool(rows[index]["p_holm"] < .05)
    for row in rows:
        row.setdefault("p_holm", None)
        row.setdefault("significant_holm", None)
        row["alpha"] = .05
        row["holm_family_size"] = family_size


def _test_summary(records: list[dict[str, Any]], ref: dict[str, Any], comp: dict[str, Any],
                  metric: str, family: str) -> dict[str, Any]:
    ref_map = {int(r["seed"]): r for r in records if r["architecture"] == ref["architecture"] and r["mode"] == ref["mode"] and r["status"] == "completed"}
    comp_map = {int(r["seed"]): r for r in records if r["architecture"] == comp["architecture"] and r["mode"] == comp["mode"] and r["status"] == "completed"}
    seeds = sorted(set(ref_map) & set(comp_map))
    expected = (42, 123, 456, 789, 1024)
    row = {"family_id": family, "reference": _pair_key(ref["architecture"], ref["mode"]),
           "competitor": _pair_key(comp["architecture"], comp["mode"]), "metric": metric,
           "seed_ids": ",".join(map(str,seeds)), "paired_n": len(seeds), "mean_difference_reference_minus_competitor": None,
           "t_statistic": None, "p_value": None, "status": "incomplete_pairs" if seeds != list(expected) else "undefined_zero_variance"}
    if seeds != list(expected):
        return row
    ref_vals = np.asarray([float(ref_map[s][metric]) for s in seeds])
    comp_vals = np.asarray([float(comp_map[s][metric]) for s in seeds])
    diff = ref_vals-comp_vals
    row["mean_difference_reference_minus_competitor"] = float(diff.mean())
    if not np.isfinite(diff).all() or not np.isfinite(ref_vals).all() or not np.isfinite(comp_vals).all():
        row["status"] = "undefined_nonfinite"
        return row
    if np.std(diff, ddof=1) == 0:
        return row
    from scipy.stats import ttest_rel
    result = ttest_rel(ref_vals, comp_vals, alternative="two-sided")
    row.update({"t_statistic": float(result.statistic), "p_value": float(result.pvalue), "status": "ok"})
    return row


def _summarize(args: argparse.Namespace, out: Path, records: list[dict[str, Any]], selection: dict[str, Any]) -> None:
    from shrine_classifier.analysis import save_csv
    records=_latest_completed_seed_rows(records)
    summary, per_class, primary, secondary = [], [], [], []
    pairs = [{"architecture": a, "mode": m} for a in args.architectures for m in args.modes]
    for pair in pairs:
        rows = [r for r in records if r["architecture"] == pair["architecture"] and r["mode"] == pair["mode"] and r.get("status") == "completed"]
        entry = {**pair, "completed_seeds": len(rows)}
        for metric in ("test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1"):
            vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
            entry[metric+"_mean"] = float(np.mean(vals)) if vals else None
            entry[metric+"_sample_sd"] = float(np.std(vals,ddof=1)) if len(vals)>=2 else None
        summary.append(entry)
        for cls in CLASSES:
            vals=[r["per_class"][cls] for r in rows if r.get("per_class",{}).get(cls)]
            pc={**pair,"class":cls,"support":vals[0]["support"] if vals else None}
            for metric in ("precision","recall","f1"):
                numbers=[float(x[metric]) for x in vals]
                pc[metric+"_mean"]=float(np.mean(numbers)) if numbers else None
                pc[metric+"_sample_sd"]=float(np.std(numbers,ddof=1)) if len(numbers)>=2 else None
            per_class.append(pc)
    ref=selection["reference_pair"]
    others=[p for p in pairs if p!=ref]
    for pair in others:
        for metric in ("test_acc","test_macro_f1"):
            primary.append(_test_summary(records,ref,pair,metric,"primary_reference_pair_28"))
    for arch in args.architectures:
        mode_pairs=[{"architecture":arch,"mode":m} for m in args.modes]
        for a,b in combinations(mode_pairs,2):
            # Fixed direction follows user-specified mode order.
            for metric in ("test_acc","test_macro_f1"):
                secondary.append(_test_summary(records,a,b,metric,"secondary_within_arch_mode_30"))
    _holm(primary, max(0, (len(args.architectures)*len(args.modes)-1)*2))
    _holm(secondary, len(args.architectures)*len(args.modes)*(len(args.modes)-1))
    save_csv(out/"final"/"pair_summary.csv",summary)
    save_csv(out/"final"/"per_class_summary.csv",per_class)
    save_csv(out/"final"/"statistical_tests.csv",primary+secondary)
    _json(out/"final"/"architecture_mode_summary.json",summary)
    _plot_summary(summary,out)


def _plot_summary(summary: list[dict[str, Any]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir=out/"report_assets"; outdir.mkdir(parents=True,exist_ok=True)
    labels=[f"{r['architecture']}\n{r['mode']}" for r in summary]
    for metric,title,name in (("test_macro_f1","Test Macro F1","macro_f1"),("test_acc","Test accuracy","accuracy")):
        vals=[r[metric+"_mean"] for r in summary]; sd=[r[metric+"_sample_sd"] or 0 for r in summary]
        fig,ax=plt.subplots(figsize=(13,5)); x=np.arange(len(labels))
        ax.bar(x,[v or 0 for v in vals],yerr=sd,capsize=3,color=["#167d9a" if r["mode"]=="full" else "#e69542" if r["mode"]=="frozen" else "#7c70b5" for r in summary])
        ax.set_xticks(x,labels,rotation=50,ha="right",fontsize=8); ax.set_ylim(0,1); ax.set_ylabel(title); ax.set_title(f"{title}: mean ± sample SD across matched seeds"); ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(outdir/f"pair_{name}.png",dpi=170); plt.close(fig)


def _confidence(args: argparse.Namespace, out: Path, records: list[dict[str,Any]]) -> dict[str,Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    confidence_dir=out/"confidence"; confidence_dir.mkdir(parents=True,exist_ok=True)
    records=_latest_completed_seed_rows(records)
    run_rows=[]; bin_rows=[]; rep_seed=42 if 42 in args.seeds else args.seeds[0]
    for record in records:
        if record.get("status")!="completed": continue
        path=Path(record["run_dir"])/"test_predictions.csv"
        if not path.is_file(): continue
        predictions=_read_csv(path)
        conf=np.asarray([float(r["confidence"]) for r in predictions]); correct=np.asarray([r["true_label"]==r["predicted_label"] for r in predictions],dtype=float)
        edges=np.linspace(0.,1.,11); bins=[]; ece=0.
        for b in range(10):
            mask=(conf>=edges[b]) & ((conf<edges[b+1]) if b<9 else (conf<=edges[b+1]))
            n=int(mask.sum()); avg=float(conf[mask].mean()) if n else None; acc=float(correct[mask].mean()) if n else None
            if n: ece+=(n/len(conf))*abs(acc-avg)
            item={"architecture":record["architecture"],"mode":record["mode"],"seed":record["seed"],"bin":b,"lower":edges[b],"upper":edges[b+1],"count":n,"mean_confidence":avg,"accuracy":acc}
            bins.append(item); bin_rows.append(item)
        row={"architecture":record["architecture"],"mode":record["mode"],"seed":record["seed"],"run_identity":record.get("identity"),
             "test_ece_10bin":float(ece),"correct_mean_confidence":float(conf[correct.astype(bool)].mean()) if correct.any() else None,
             "incorrect_mean_confidence":float(conf[~correct.astype(bool)].mean()) if (~correct.astype(bool)).any() else None,"test_n":len(predictions),"status":"completed"}
        run_rows.append(row)
        if int(record["seed"])==rep_seed:
            points=[x for x in bins if x["count"]]
            fig,ax=plt.subplots(figsize=(5,5));ax.plot([0,1],[0,1],"--",color="gray",label="ideal")
            if points:ax.plot([x["mean_confidence"] for x in points],[x["accuracy"] for x in points],"o-",label=f"ECE={ece:.3f}")
            ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean confidence",ylabel="Accuracy",title=f"{record['architecture']} · {record['mode']} · seed {rep_seed}")
            ax.grid(alpha=.2);ax.legend();fig.tight_layout();dest=confidence_dir/f"reliability_{record['architecture']}_{record['mode']}_seed{rep_seed}.png";fig.savefig(dest,dpi=150);plt.close(fig)
            fig,ax=plt.subplots(figsize=(6,4))
            if correct.any():ax.hist(conf[correct.astype(bool)],bins=np.linspace(0,1,11),alpha=.65,label="Correct")
            if (~correct.astype(bool)).any():ax.hist(conf[~correct.astype(bool)],bins=np.linspace(0,1,11),alpha=.65,label="Incorrect")
            ax.set(xlim=(0,1),xlabel="Top-class confidence",ylabel="Test image count",title=f"Confidence · {record['architecture']} · {record['mode']} · seed {rep_seed}")
            ax.grid(axis="y",alpha=.2);ax.legend();fig.tight_layout();fig.savefig(confidence_dir/f"confidence_hist_{record['architecture']}_{record['mode']}_seed{rep_seed}.png",dpi=150);plt.close(fig)
    _csv(confidence_dir/"run_metrics.csv",run_rows);_csv(confidence_dir/"bin_metrics.csv",bin_rows)
    summary=[]
    for arch in args.architectures:
        for mode in args.modes:
            vals=[r["test_ece_10bin"] for r in run_rows if r["architecture"]==arch and r["mode"]==mode]
            summary.append({"architecture":arch,"mode":mode,"completed_seeds":len(vals),"test_ece_10bin_mean":float(np.mean(vals)) if vals else None,
                            "test_ece_10bin_sample_sd":float(np.std(vals,ddof=1)) if len(vals)>=2 else None})
    _csv(confidence_dir/"pair_summary.csv",summary)
    return {"runs":len(run_rows),"bins":10}


def _analyze(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    registry=_load_registry(out); selection=json.loads((out/"selection.json").read_text(encoding="utf-8"))
    records=registry["final_runs"]
    _summarize(args,out,records,selection)
    confidence=_confidence(args,out,records)
    _logged_curves(args,out,records)
    dynamics=_training_dynamics(args,out,records)
    _tuning_plot(out,registry["trials"])
    return {"run_attempts":len(records),"completed_unique_seeds":len(_latest_completed_seed_rows(records)),"confidence":confidence,"dynamics":dynamics}


def _logged_curves(args: argparse.Namespace, out: Path, records: list[dict[str,Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for record in _latest_completed_seed_rows(records):
        if record.get("status")!="completed" or int(record["seed"]) != (42 if 42 in args.seeds else args.seeds[0]): continue
        path=Path(record["run_dir"])/"history.csv"
        if not path.exists(): continue
        hist=[r for r in _read_csv(path) if r.get("train_loss") not in (None,"") and r.get("val_loss") not in (None,"")]
        if not hist: continue
        fig,axes=plt.subplots(1,3,figsize=(11,3.3))
        for ax,keys,title in zip(axes,(('train_loss','val_loss'),('train_acc','val_acc'),('train_macro_f1','val_macro_f1')),('Loss','Accuracy','Macro F1')):
            for key in keys: ax.plot([int(h['epoch']) for h in hist],[float(h[key]) for h in hist],label=key)
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=7)
        fig.suptitle(f"{record['architecture']} · {record['mode']} · seed {record['seed']}"); fig.tight_layout()
        p=out/'final'/'curves'/record['architecture']/record['mode']/f"seed{record['seed']}.png"; p.parent.mkdir(parents=True,exist_ok=True); fig.savefig(p,dpi=150); plt.close(fig)


def _tuning_plot(out: Path, rows: list[dict[str,Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values=[r for r in rows if r.get('status')=='completed' and r.get('best_val_macro_f1') is not None]
    if not values:return
    fig,ax=plt.subplots(figsize=(8,5))
    for mode,marker in zip(MODES,('o','s','^')):
        subset=[r for r in values if r['mode']==mode]
        ax.scatter([r['config']['learning_rate'] for r in subset],[r['best_val_macro_f1'] for r in subset],label=mode,marker=marker,alpha=.7)
    ax.set_xscale('log');ax.set_xlabel('Learning rate');ax.set_ylabel('Best validation Macro F1');ax.set_title('Logged search trials (train/validation only)');ax.grid(alpha=.2);ax.legend()
    p=out/'tuning'/'tuning_validation_scatter.png';p.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(p,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for mode,marker in zip(MODES,('o','s','^')):
        subset=[r for r in values if r['mode']==mode]
        ax.scatter([r['config']['weight_decay'] for r in subset],[r['best_val_macro_f1'] for r in subset],label=mode,marker=marker,alpha=.7)
    ax.set_xscale('log');ax.set_xlabel('Weight decay');ax.set_ylabel('Best validation Macro F1');ax.set_title('Logged trial weight decay vs validation score');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'tuning'/'weight_decay_validation_scatter.png',dpi=160);plt.close(fig)


def _training_dynamics(args: argparse.Namespace, out: Path, records: list[dict[str,Any]]) -> dict[str,Any]:
    rows=[]
    for record in _latest_completed_seed_rows(records):
        path=Path(record["run_dir"])/"history.csv"
        if not path.is_file():continue
        history=[r for r in _read_csv(path) if r.get("train_loss") not in (None,"") and r.get("val_loss") not in (None,"")]
        best=next((r for r in history if int(r["epoch"])==int(record["best_epoch"])),None)
        if best is None:continue
        rows.append({"architecture":record["architecture"],"mode":record["mode"],"seed":record["seed"],
            "run_identity":record.get("identity"),"best_epoch":record["best_epoch"],"stop_epoch":record.get("epochs_completed"),
            "train_val_loss_gap_at_best":float(best["val_loss"])-float(best["train_loss"]),
            "train_val_accuracy_gap_at_best":float(best["train_acc"])-float(best["val_acc"]),
            "train_val_macro_f1_gap_at_best":float(best["train_macro_f1"])-float(best["val_macro_f1"]),"status":"completed"})
    dest=out/"diagnostics";dest.mkdir(parents=True,exist_ok=True);_csv(dest/"run_dynamics.csv",rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if rows:
        labels=[];loss_gaps=[];f1_gaps=[]
        for arch in args.architectures:
            for mode in args.modes:
                group=[r for r in rows if r["architecture"]==arch and r["mode"]==mode]
                labels.append(f"{arch}\n{mode}");loss_gaps.append(float(np.mean([r["train_val_loss_gap_at_best"] for r in group])) if group else 0)
                f1_gaps.append(float(np.mean([r["train_val_macro_f1_gap_at_best"] for r in group])) if group else 0)
        fig,axes=plt.subplots(1,2,figsize=(14,5));x=np.arange(len(labels))
        axes[0].bar(x,loss_gaps);axes[0].set_title("Validation − train loss at selected checkpoint");axes[0].set_ylabel("Loss gap")
        axes[1].bar(x,f1_gaps);axes[1].set_title("Train − validation Macro F1 at selected checkpoint");axes[1].set_ylabel("Macro F1 gap")
        for ax in axes:ax.set_xticks(x,labels,rotation=55,ha="right",fontsize=7);ax.grid(axis="y",alpha=.2)
        fig.tight_layout();fig.savefig(dest/"training_gaps.png",dpi=160);plt.close(fig)
    return {"run_rows":len(rows),"gap_figure":str(dest/"training_gaps.png") if rows else None}


def _eyeball_and_cam(args: argparse.Namespace, out: Path) -> dict[str,Any]:
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torchvision import transforms, models
    from shrine_classifier.data import open_rgb, letterbox, IMAGENET_MEAN, IMAGENET_STD
    from shrine_classifier.models import build_model
    from shrine_classifier.analysis import gradcam
    selection,winners=_verify_lock(args,out); ref=selection['reference_pair']; seed=selection['representative_seed']
    row=next(r for r in _latest_completed_seed_rows(_load_registry(out)['final_runs']) if r.get('architecture')==ref['architecture'] and r.get('mode')==ref['mode'] and int(r.get('seed',-1))==seed)
    preds=_read_csv(Path(row['run_dir'])/'test_predictions.csv')
    chosen=[]
    for cls in CLASSES:
        subset=[p for p in preds if p['true_label']==cls]
        for right in (True,False):
            found=[p for p in subset if (p['predicted_label']==cls)==right]
            if found: chosen.append(sorted(found,key=lambda p:(float(p['confidence']),p['image_id']) if not right else (-float(p['confidence']),p['image_id']))[0])
            elif subset:
                candidate=min(subset,key=lambda p:(float(p['confidence']),p['image_id']))
                if candidate not in chosen: chosen.append(candidate)
    if not chosen: raise RuntimeError('No test predictions available for CAM/eyeball')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    # The ImageNet output space remains 1,000 classes; save its generic top label
    # for the exact same images, and never calculate shrine-class metrics from it.
    enum_name={'resnet50':'ResNet50_Weights','efficientnet_b0':'EfficientNet_B0_Weights','mobilenet_v3_large':'MobileNet_V3_Large_Weights','densenet121':'DenseNet121_Weights','convnext_tiny':'ConvNeXt_Tiny_Weights'}[ref['architecture']]
    weights=getattr(models,enum_name).DEFAULT; imagenet_model=getattr(models,ref['architecture'])(weights=weights).to(device).eval(); image_transform=weights.transforms()
    split_lookup={r['image_id']:r for r in _read_csv(out/'splits.csv')}
    imagenet_top1={}
    with torch.no_grad():
        for item in chosen:
            image=open_rgb(args.data_root/split_lookup[item['image_id']]['path'])
            logits=imagenet_model(image_transform(image).unsqueeze(0).to(device)); probs=torch.softmax(logits,dim=1)[0]
            score,index=probs.max(0);imagenet_top1[item['image_id']]={'imagenet_label':weights.meta['categories'][int(index)],'imagenet_confidence':float(score)}
    del imagenet_model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    model,meta=build_model(ref['architecture'],pretrained=False); state=torch.load(row['checkpoint'],map_location=device,weights_only=False); model.load_state_dict(state['model_state']);model.to(device).eval()
    transforms_tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
    cam_dir=out/'gradcam';cam_dir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(len(chosen),3,figsize=(10,3.1*len(chosen)),squeeze=False)
    summary=[]
    for i,item in enumerate(chosen):
        rec=next(r for r in _read_csv(out/'splits.csv') if r['image_id']==item['image_id'])
        image=open_rgb(args.data_root/rec['path']);canvas,bbox=letterbox(image,args.input_size);tensor=transforms_tensor(canvas).unsqueeze(0).to(device)
        idx=CLASSES.index(item['predicted_label'])
        cam=gradcam(model,ref['architecture'],tensor,idx)[0].detach().cpu().numpy()
        x1,y1,x2,y2=bbox;cam=Image.fromarray(np.uint8(np.clip(cam[y1:y2,x1:x2],0,1)*255)).resize(image.size,Image.Resampling.BILINEAR)
        heat=np.asarray(cam,dtype=np.float32)/255
        base=np.asarray(image,dtype=np.float32); colored=plt.get_cmap('jet')(heat)[...,:3]*255
        overlay=np.uint8(np.clip(base*.55+colored*.45,0,255))
        axes[i,0].imshow(image);axes[i,0].set_title(f"Original · true {item['true_label']}")
        axes[i,1].imshow(heat,cmap='jet');axes[i,1].set_title(f"Grad-CAM target: {item['predicted_label']}")
        axes[i,2].imshow(overlay);axes[i,2].set_title(f"Pred {item['predicted_label']} · conf {float(item['confidence']):.2f}")
        for ax in axes[i]:ax.axis('off')
        sample_dir=cam_dir/'samples'/item['image_id'].replace('/','__');sample_dir.mkdir(parents=True,exist_ok=True)
        image.save(sample_dir/'original.png');cam.save(sample_dir/'heatmap.png');Image.fromarray(overlay).save(sample_dir/'overlay.png')
        summary.append({**item,**imagenet_top1[item['image_id']],'reason':'correct case' if item['true_label']==item['predicted_label'] else 'misclassified case', 'explained_target':item['predicted_label'],'checkpoint':row['checkpoint'],'cam_files':str(sample_dir.relative_to(out))})
    fig.tight_layout();fig.savefig(cam_dir/'reference_pair_cam_grid.png',dpi=150);plt.close(fig)
    _csv(cam_dir/'selected_cases.csv',summary);_csv(out/'eyeball'/'eyeball_cases.csv',summary)
    fig,axes=plt.subplots(len(chosen),2,figsize=(8,3*len(chosen)),squeeze=False)
    for i,item in enumerate(chosen):
        rec=next(r for r in _read_csv(out/'splits.csv') if r['image_id']==item['image_id']);image=open_rgb(args.data_root/rec['path'])
        axes[i,0].imshow(image);axes[i,0].set_title(f"True: {item['true_label']} · {item['image_id']}")
        axes[i,1].text(0,.75,f"ImageNet top-1: {summary[i]['imagenet_label']} ({summary[i]['imagenet_confidence']:.2f})\nFine-tuned: {item['predicted_label']} ({float(item['confidence']):.2f})\nCorrect: {item['true_label']==item['predicted_label']}\nReason: {summary[i]['reason']}",fontsize=11);axes[i,1].axis('off');axes[i,0].axis('off')
    fig.tight_layout();(out/'eyeball').mkdir(parents=True,exist_ok=True);fig.savefig(out/'eyeball'/'eyeball_grid.png',dpi=150);plt.close(fig)
    return {'reference_pair':ref,'seed':seed,'samples':len(chosen)}


def _report(args: argparse.Namespace, out: Path) -> dict[str,Any]:
    summary=_read_csv(out/'final'/'pair_summary.csv')
    tests=_read_csv(out/'final'/'statistical_tests.csv')
    selection=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    all_registry=_load_registry(out)
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    tuning_target=len(protocol['architectures'])*len(protocol['modes'])*int(protocol['trials_per_pair'])
    final_target=len(protocol['architectures'])*len(protocol['modes'])*len(protocol['final_seeds'])
    seeds_per_pair=len(protocol['final_seeds'])
    completed=len(_latest_completed_seed_rows(all_registry['final_runs']))
    completed_trials=len({r.get('identity') for r in all_registry['trials'] if r.get('status')=='completed'})
    winners=json.loads((out/'tuning'/'best_params_by_arch_mode.json').read_text(encoding='utf-8'))['pairs']
    recipe_rows=''.join(f"<tr><td>{html.escape(v['architecture'])}</td><td>{html.escape(v['mode'])}</td><td>{float(v['learning_rate']):.3g}</td><td>{float(v['weight_decay']):.3g}</td><td>{html.escape(v['optimizer'])}</td><td>{v['batch_size']}</td><td>{v['winning_trial']}</td></tr>" for v in winners.values())
    source=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
    sections=[]
    for row in summary:
        f1=row.get('test_macro_f1_mean');sd=row.get('test_macro_f1_sample_sd')
        value = f"{float(f1):.3f} ± {float(sd):.3f}" if f1 not in (None, "", "nan") and sd not in (None, "", "nan") else "incomplete"
        sections.append(f"<tr><td>{html.escape(str(row['architecture']))}</td><td>{html.escape(str(row['mode']))}</td><td>{int(row['completed_seeds'])}/{seeds_per_pair}</td><td>{value}</td></tr>")
    figure_sources=[(p,p.name) for p in sorted((out/'report_assets').glob('*.png'))]
    for rel in ('dataset_audit/class_distribution.png','dataset_audit/sample_grid.png','dataset_audit/preprocessing_examples.png',
                'dataset_audit/near_duplicate_review.png','baseline/pretrained_examples.png',
                'gradcam/reference_pair_cam_grid.png','eyeball/eyeball_grid.png'):
        p=out/rel
        if p.is_file():figure_sources.append((p,'../'+rel.replace('\\','/')))
    reference=selection['reference_pair'];representative=selection['representative_seed']
    for rel in (f"confidence/reliability_{reference['architecture']}_{reference['mode']}_seed{representative}.png",
                f"confidence/confidence_hist_{reference['architecture']}_{reference['mode']}_seed{representative}.png"):
        p=out/rel
        if p.is_file():figure_sources.append((p,'../'+rel))
    for rel in ('tuning/tuning_validation_scatter.png','tuning/weight_decay_validation_scatter.png','diagnostics/training_gaps.png',
                f"final/curves/{reference['architecture']}/{reference['mode']}/seed{representative}.png"):
        p=out/rel
        if p.is_file():figure_sources.append((p,'../'+rel))
    figures=''.join(f"<figure><img src='{src}'><figcaption>{html.escape(p.name)}</figcaption></figure>" for p,src in figure_sources)
    counts={}
    for row in tests:
        counts[row.get('status','unknown')]=counts.get(row.get('status','unknown'),0)+1
    body=f"""<!doctype html><html lang='th'><meta charset='utf-8'><title>สรุปการทดลอง Thai Shrine</title><style>body{{font-family:system-ui,'Noto Sans Thai',sans-serif;max-width:1100px;margin:2rem auto;line-height:1.6}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #aaa;padding:.45rem;text-align:left}}th{{background:#17324d;color:white}}img{{max-width:100%}}figure{{margin:2rem 0}}code{{background:#eee;padding:.15rem}}</style><h1>อธิบายผลการทดลอง Thai Shrine Classifier</h1>
<p>เอกสารนี้สร้างจาก logs ภายใต้ <code>{html.escape(out.name)}</code> เท่านั้น งานที่บันทึกเสร็จ: tuning สำเร็จ {completed_trials}/{tuning_target} unique trials; final {completed}/{final_target} unique seeds.</p>
<h2>ข้อมูลและ preprocessing</h2><p>ใช้ภาพ {source['image_count']} ภาพใน 4 classes และ canonical grouped split เดิม train 390 / validation 85 / test 85 จาก 550 groups. URL ไม่ทราบ {source['unknown_url_count']} ภาพ. แปลงภาพเป็น RGB, EXIF orientation, resize โดยรักษาสัดส่วนลง canvas 224 × 224 สี neutral RGB (124,116,104), แล้ว normalize แบบ ImageNet. Augmentation ใช้ train เท่านั้น.</p>
<h2>คำถามที่ทดลอง</h2><p>เปรียบเทียบ CNN 5 แบบและนโยบายการฝึก 3 แบบ โดยให้แต่ละคู่สุ่มค้น learning rate, weight decay, optimizer และ batch size จาก validation ของตัวเอง ดังนั้น Full, Frozen และ Partial มี hyperparameters ชนะของตัวเอง. ใช้ ImageNet weights, class-weighted cross entropy, AdamW/Adam/SGD และ ReduceLROnPlateau. Frozen ฝึก classifier; Partial เปิด semantic final stage กับ classifier; Full ฝึก backbone ทั้งหมด.</p>
<h2>ขั้นตอนและการอ่านผล</h2><ol><li>ตรวจไฟล์/provenance และใช้ grouped split เดิม 70/15/15 เพื่อกัน duplicate group ข้าม split</li><li>สุ่ม {int(protocol['trials_per_pair'])} configurations ต่อ architecture × mode ด้วย seed {int(protocol['tuning_seed'])}; trials ใช้ train/validation เท่านั้น คะแนนในกราฟ tuning คือผล validation ไม่ใช่ test</li><li>เลือกสูตรชนะต่อคู่ตาม best validation Macro F1, Accuracy, lower loss, trial order แล้วล็อก reference pair จาก {len(protocol['architectures'])*len(protocol['modes'])} คู่ด้วย validation</li><li>ฝึกจาก ImageNet weights ใหม่ {seeds_per_pair} matched seeds ต่อคู่ เลือก checkpoint ด้วย validation แล้วประเมิน test ภายหลัง</li><li>สรุป mean ± sample SD จาก seeds; paired tests join ด้วย seed ID และปรับ Holm แยก primary family 28 และ secondary family 30</li></ol>
<h2>Validation-selected reference</h2><p><b>{html.escape(selection['reference_pair']['architecture'])} / {html.escape(selection['reference_pair']['mode'])}</b>; selection ไม่ได้ใช้ test scores. test split เคยเปิดดูในงานก่อนหน้า การทดลองใหม่นี้จึงเป็น exploratory บน split เดิม.</p>
<h2>ผลแยกตามคู่</h2><table><thead><tr><th>Architecture</th><th>Mode</th><th>Seeds</th><th>Test Macro F1 mean ± sample SD</th></tr></thead><tbody>{''.join(sections)}</tbody></table>
<h2>สูตรที่ชนะจาก validation ต่อคู่</h2><table><thead><tr><th>Architecture</th><th>Mode</th><th>Learning rate</th><th>Weight decay</th><th>Optimizer</th><th>Batch</th><th>Trial</th></tr></thead><tbody>{recipe_rows}</tbody></table>
<p>ผล class-specific อยู่ใน <code>final/per_class_summary.csv</code>. Confusion matrix มีทั้งจำนวนภาพและสัดส่วนต่อ true class ต่อ run.</p><p>Calibration ใช้ Expected Calibration Error 10 bins จาก confidence ของ predictions หลัง final; ตาราง/รูปแยก architecture × mode × seed อยู่ใน <code>confidence/</code>.</p>
<p>สถานะ paired tests: {html.escape(json.dumps(counts,ensure_ascii=False))}. อ่านค่า effect size, p และ adjusted p ใน <code>final/statistical_tests.csv</code>; ค่า undefined/incomplete ไม่ใช่ผลไม่มีนัยสำคัญ.</p>
<h2>Figures จาก log</h2>{figures or '<p>ยังไม่มีกราฟจากผล final ที่บันทึก</p>'}
<h2>ข้อจำกัด</h2><ul><li>ผลต่างระหว่าง seeds สะท้อนความแปรปรวนบน split เดียว ไม่แทนชุดข้อมูลใหม่</li><li>การทำ CAM เป็นเครื่องมือวินิจฉัย ไม่ยืนยันเหตุและผล</li><li>อ่านค่า W&amp;B ตามสถานะจริงของแต่ละ run; การมีไฟล์ local ไม่ยืนยันว่า sync ขึ้น cloud</li><li>แนวทางผู้สอนและ PDF อ้างอิงไม่มีในโฟลเดอร์นี้ จึงยังยืนยันความสอดคล้องกับผู้สอนไม่ได้</li></ul>
<h2>แหล่งข้อมูล</h2><p>metrics: <code>final/run_metrics.csv</code>; configs/history: ไดเรกทอรี run ใน <code>tuning/</code> และ <code>final/</code>; definitions: <code>protocol.json</code>, <code>selection.json</code>; pair summaries, class metrics, statistical tests และรูปภาพอยู่ใต้ out-dir เดียวกัน.</p></html>"""
    p=out/'report_assets'/'understanding_th.html';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body,encoding='utf-8')
    return {'path':str(p),'final_runs':completed,'tuning_trials':len(_load_registry(out)['trials'])}


def _smoke(args: argparse.Namespace, out: Path) -> dict[str,Any]:
    import torch
    from shrine_classifier.models import build_model
    from shrine_classifier_ext.finetuning_modes import (apply_finetuning_mode,enforce_mode_per_epoch,
        _get_head_modules,_get_partial_backbone_modules)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    results=[]
    for arch in args.architectures:
        for mode in args.modes:
            model,meta=build_model(arch,pretrained=False);model.to(device);stats=apply_finetuning_mode(model,arch,mode)
            parameters=[p for p in model.parameters() if p.requires_grad]
            all_parameters=list(model.parameters())
            expected_ids={id(p) for p in all_parameters} if mode=='full' else {id(p) for m in _get_head_modules(model,arch) for p in m.parameters()}
            if mode=='partial':expected_ids.update(id(p) for m in _get_partial_backbone_modules(model,arch) for p in m.parameters())
            actual_ids={id(p) for p in all_parameters if p.requires_grad}
            optimizer=torch.optim.SGD(parameters,lr=1e-4)
            model.train();enforce_mode_per_epoch(model,arch,mode)
            x=torch.randn(2,3,args.input_size,args.input_size,device=device);y=torch.tensor([0,1],device=device)
            before={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
            before_trainable={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
            frozen_bn_buffers={}
            for module_name,module in model.named_modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) and not module.training:
                    for buffer_name,value in module.named_buffers(recurse=False):
                        frozen_bn_buffers[f'{module_name}.{buffer_name}']=value.detach().clone()
            loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step()
            unchanged=all(torch.equal(v,dict(model.named_parameters())[n]) for n,v in before.items())
            current_params=dict(model.named_parameters())
            updated=any(not torch.equal(value,current_params[name]) for name,value in before_trainable.items())
            current_buffers=dict(model.named_buffers())
            bn_stable=all(torch.equal(value,current_buffers[name]) for name,value in frozen_bn_buffers.items())
            no_frozen_grad=all(current_params[name].grad is None for name in before)
            passed=(actual_ids==expected_ids and unchanged and updated and bn_stable and no_frozen_grad and loss.isfinite().item())
            row={'architecture':arch,'mode':mode,'logits':[2,4],'trainable_parameters':stats['trainable_parameters'],'frozen_parameters':stats['frozen_parameters'],
                 'trainable_membership_verified':actual_ids==expected_ids,'frozen_unchanged':unchanged,'trainable_parameter_updated':updated,
                 'frozen_bn_buffers_stable':bn_stable,'frozen_gradient_absent':no_frozen_grad,'pretrained':False,'test_split_accessed':False,'status':'passed' if passed else 'failed'}
            results.append(row);del model,optimizer,x,loss
            if torch.cuda.is_available():torch.cuda.empty_cache()
            if row['status']!='passed':raise RuntimeError(f"Mode smoke failed: {row}")
    _json(out/'smoke'/'mode_smoke.json',{'models':results,'status':'passed','test_split_accessed':False,'measured_training':False,'weights':'synthetic random; not pretrained'})
    return {'models':len(results),'status':'passed','test_split_accessed':False}


def _parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('prepare','smoke','tune','lock','final','analyze','gradcam','eyeball','report','all'))
    p.add_argument('--data-root',type=Path,default=Path('dataset_cleaned'));p.add_argument('--padded-root',type=Path,default=Path('dataset_padding'))
    p.add_argument('--base-out-dir',type=Path,default=Path('outputs'));p.add_argument('--out-dir',type=Path,default=Path('outputs_modes_v3'))
    p.add_argument('--architectures',nargs='+',choices=ARCHITECTURES,default=list(ARCHITECTURES));p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--trials-per-pair','--trials',dest='trials_per_pair',type=int,default=5);p.add_argument('--runs',type=int,default=5);p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--input-size',type=int,default=224);p.add_argument('--seed',type=int,default=42);p.add_argument('--split-seed',type=int,default=42)
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto');p.add_argument('--num-workers',type=int,default=0)
    p.add_argument('--wandb-project',default='thai-shrine-classifier');p.add_argument('--wandb-entity',default=None);p.add_argument('--wandb-mode',choices=('online','offline','disabled'),default='online')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    return p


def main(argv: list[str] | None = None) -> int:
    args=_parser().parse_args(argv)
    args.out_dir=args.out_dir.resolve();args.data_root=args.data_root.resolve();args.base_out_dir=args.base_out_dir.resolve();args.padded_root=args.padded_root.resolve()
    if args.runs!=len(args.seeds) or len(set(args.seeds))!=len(args.seeds):raise ValueError('--runs must equal count of unique --seeds')
    if args.trials_per_pair<1 or args.epochs<1 or not args.architectures or not args.modes:raise ValueError('architecture/mode/trial/epoch selections must be non-empty and positive')
    if args.dry_run:
        print(json.dumps({'stage':args.stage,'out_dir':str(args.out_dir),'architectures':args.architectures,'modes':args.modes,'trials_per_pair':args.trials_per_pair,'tuning_runs':len(args.architectures)*len(args.modes)*args.trials_per_pair,'final_runs':len(args.architectures)*len(args.modes)*args.runs,'max_epochs':len(args.architectures)*len(args.modes)*(args.trials_per_pair+args.runs)*args.epochs,'test_access_during_tune_or_lock':False,'measured_run_started':False},ensure_ascii=False,indent=2));return 0
    args.out_dir.mkdir(parents=True,exist_ok=True);(args.out_dir/'wandb').mkdir(exist_ok=True)
    os.environ['WANDB_DIR']=str(args.out_dir/'wandb');os.environ.setdefault('WANDB_INIT_TIMEOUT','15');os.environ.setdefault('WANDB_START_METHOD','thread')
    stages=('prepare','smoke','tune','lock','final','analyze','gradcam','report') if args.stage=='all' else (args.stage,)
    for stage in stages:
        start=time.time()
        if stage=='prepare': result=_verify_and_prepare(args,args.out_dir)
        elif stage=='smoke':result=_smoke(args,args.out_dir)
        elif stage=='tune':result=_tune(args,args.out_dir)
        elif stage=='lock':result=_lock(args,args.out_dir)
        elif stage=='final':result=_final(args,args.out_dir)
        elif stage=='analyze':result=_analyze(args,args.out_dir)
        elif stage in ('gradcam','eyeball'):result=_eyeball_and_cam(args,args.out_dir)
        elif stage=='report':result=_report(args,args.out_dir)
        print(f"[{stage}] {json.dumps(result,ensure_ascii=False,default=str)[:1200]} elapsed={time.time()-start:.1f}s",flush=True)
    return 0


if __name__=='__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())

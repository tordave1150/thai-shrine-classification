from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import paired_holm_tests, save_csv, save_json, summarize_runs
from .data import CLASSES
from .models import ARCHITECTURES
from .training import fit_run, load_split

SCREEN = {"optimizer": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4,
          "batch_size": 16, "epochs": 30, "patience": 7, "input_size": 224}
SEARCH_SPACE = {
    "method": "random", "metric": {"name": "best_val_macro_f1", "goal": "maximize"},
    "parameters": {"learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
                   "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
                   "optimizer": {"values": ["adamw", "adam", "sgd"]},
                   "batch_size": {"values": [16, 32]}},
}


def default_config(args) -> dict[str, Any]:
    return {**SCREEN, "epochs": int(args.epochs), "input_size": int(args.input_size),
            "black_threshold": int(args.black_threshold)}


def final_seed_list(args) -> list[int]:
    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        run_count = args.runs or 5
        base = [42, 123, 456, 789, 1024]
        seeds = base[:min(run_count, len(base))]
        rng = np.random.default_rng(args.seed)
        while len(seeds) < run_count:
            candidate = int(rng.integers(10000, 2**31 - 1))
            if candidate not in seeds:
                seeds.append(candidate)
    if not 3 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds):
        raise ValueError("Final run seeds must contain 3-10 distinct integers")
    if args.runs is not None and args.runs != len(seeds):
        raise ValueError(f"--runs={args.runs} disagrees with {len(seeds)} seed values")
    return seeds


def _append_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            current = list(csv.DictReader(stream))
    current.append(result)
    save_csv(path, current)


def run_screen(args, out: Path) -> list[dict[str, Any]]:
    split = out / "splits.csv"
    rows = load_split(split)
    results = []
    for arch in args.architectures:
        try:
            result = fit_run(root=args.data_root, split_csv=split, out=out, architecture=arch,
                             seed=args.seed, config={**default_config(args), "augmentation": True,
                             "use_class_weights": True}, stage="screen", wandb_project=args.wandb_project,
                             wandb_entity=args.wandb_entity, wandb_mode=args.wandb_mode, device=args.device, workers=args.num_workers)
        except Exception as exc:
            result = {"architecture": arch, "seed": args.seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        _append_result(out / "screening" / "screening_results.csv", result)
    save_json(out / "screening" / "screening_results.json", results)
    return results


def _sample_local_configs(trials: int, seed: int):
    rng = np.random.default_rng(seed)
    optimizers = ("adamw", "adam", "sgd")
    batches = (16, 32)
    for _ in range(trials):
        yield {"learning_rate": float(10 ** rng.uniform(-5, -3)),
               "weight_decay": float(10 ** rng.uniform(-6, -2)),
               "optimizer": str(rng.choice(optimizers)), "batch_size": int(rng.choice(batches))}


def _search_objective(args, out: Path, arch: str, trial_no: int, config: dict[str, Any], active_run=None,
                      tracking_mode: str | None = None) -> dict[str, Any]:
    trial_id = f"sweep_{arch}_trial{trial_no:03d}_{int(time.time())}"
    try:
        result = fit_run(root=args.data_root, split_csv=out / "splits.csv", out=out, architecture=arch,
                         seed=args.seed, config={**default_config(args), **config, "augmentation": True,
                         "use_class_weights": True, "stage": "tune", "trial": trial_no},
                         stage="tuning", wandb_project=args.wandb_project,
                         wandb_entity=args.wandb_entity,
                         wandb_mode=tracking_mode or args.wandb_mode, device=args.device,
                         workers=args.num_workers, active_wandb_run=active_run)
        result.update({"trial": trial_no, "trial_id": trial_id, **config})
        if active_run is not None:
            active_run.summary.update({"best_epoch": result["best_epoch"],
                                       "best_val_acc": result["best_val_acc"],
                                       "best_val_macro_f1": result["best_val_macro_f1"]})
    except Exception as exc:
        result = {"architecture": arch, "trial": trial_no, "trial_id": trial_id, "status": "failed",
                  **config, "error": f"{type(exc).__name__}: {exc}"}
    if active_run is not None:
        try:
            active_run.finish(exit_code=0 if result.get("status") == "completed" else 1)
        except Exception:
            pass
    _append_result(out / "tuning" / "sweep_summary.csv", result)
    return result


def run_tuning(args, out: Path) -> dict[str, Any]:
    load_split(out / "splits.csv")  # no test loader is constructed during search
    tuning_dir = out / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    backends: dict[str, str] = {}
    sweep_meta: dict[str, Any] = {"project": args.wandb_project, "trials_per_architecture": args.trials,
                                  "common_recipe": default_config(args), "search_space": SEARCH_SPACE, "sweeps": {}}
    for ai, arch in enumerate(args.architectures):
        arch_results: list[dict[str, Any]] = []
        sweep_id = None
        wandb_module = None
        if args.wandb_mode == "online":
            try:
                import wandb as wandb_module
                config = {"method": "random", "metric": {"name": "best_val_macro_f1", "goal": "maximize"},
                          "parameters": {**SEARCH_SPACE["parameters"], "architecture": {"value": arch},
                                         "seed": {"value": args.seed}, "stage": {"value": "tune"}}}
                sweep_id = wandb_module.sweep(config, project=args.wandb_project, entity=args.wandb_entity)
                backends[arch] = "wandb_sweep_online"
            except Exception as exc:
                sweep_meta["sweeps"][arch] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
                wandb_module, sweep_id = None, None
        if sweep_id and wandb_module:
            trial_count = 0
            def objective():
                nonlocal trial_count
                trial_count += 1
                run = wandb_module.init(project=args.wandb_project, group="sweep",
                                        entity=args.wandb_entity, name=f"sweep_{arch}_trial{trial_count:03d}", reinit=True)
                params = {k: run.config[k] for k in ("learning_rate", "weight_decay", "optimizer", "batch_size")}
                arch_results.append(_search_objective(args, out, arch, trial_count, params, active_run=run, tracking_mode="online"))
            try:
                wandb_module.agent(sweep_id, function=objective, count=args.trials,
                                   project=args.wandb_project, entity=args.wandb_entity)
                sweep_meta["sweeps"][arch] = {"status": "completed", "sweep_id": sweep_id, "trials": trial_count}
            except Exception as exc:
                sweep_meta["sweeps"][arch] = {"status": "interrupted", "sweep_id": sweep_id,
                                              "completed_trials": trial_count, "error": f"{type(exc).__name__}: {exc}"}
            remaining = args.trials - len(arch_results)
            fallback_mode = "offline" if wandb_module is not None else "disabled"
            if remaining > 0:
                backends[arch] += f"+local_random_{fallback_mode}"
                for no, params in enumerate(_sample_local_configs(remaining, args.seed + ai * 1009 + trial_count), trial_count + 1):
                    arch_results.append(_search_objective(args, out, arch, no, params, tracking_mode=fallback_mode))
        else:
            available = False
            try:
                import wandb  # noqa: F401
                available = True
            except Exception:
                pass
            fallback_mode = "offline" if args.wandb_mode == "offline" or (args.wandb_mode == "online" and available) else "disabled"
            backends[arch] = f"local_random_{fallback_mode}"
            for no, params in enumerate(_sample_local_configs(args.trials, args.seed + ai * 1009), 1):
                arch_results.append(_search_objective(args, out, arch, no, params, tracking_mode=fallback_mode))
            if args.wandb_mode == "disabled":
                backends[arch] = "local_random_disabled"
        if not arch_results:
            raise RuntimeError(f"Tuning produced no recorded trials for {arch}")
        all_results.extend(arch_results)
        complete = [r for r in arch_results if r.get("status") == "completed" and r.get("best_val_macro_f1") is not None]
        if not complete:
            raise RuntimeError(f"All {len(arch_results)} tuning trials failed for {arch}")
        arch_results.sort(key=lambda r: (-float(r["best_val_macro_f1"]), -float(r.get("best_val_acc", 0)), float(r.get("best_val_loss", float("inf"))), int(r["trial"])))
        sweep_meta["sweeps"].setdefault(arch, {"status": "completed_locally"})
        sweep_meta["sweeps"][arch]["backend"] = backends[arch]
        sweep_meta["sweeps"][arch]["successful_trials"] = len(complete)
        sweep_meta["sweeps"][arch]["failed_trials"] = len(arch_results) - len(complete)
        best = arch_results[0]
        sweep_meta["sweeps"][arch]["best_trial"] = {k: best.get(k) for k in ("trial", "learning_rate", "weight_decay", "optimizer", "batch_size", "best_val_acc", "best_val_macro_f1", "best_val_loss")}

    # Fixed architecture order is a stable last-resort tie-break; no test scores are present here.
    by_arch = {}
    for arch in args.architectures:
        candidates = [r for r in all_results if r["architecture"] == arch and r.get("status") == "completed"]
        candidates.sort(key=lambda r: (-r["best_val_macro_f1"], -r["best_val_acc"], r["best_val_loss"], r["trial"]))
        best = candidates[0]
        by_arch[arch] = {"architecture": arch, "learning_rate": best["learning_rate"], "weight_decay": best["weight_decay"],
                         "optimizer": best["optimizer"], "batch_size": best["batch_size"],
                         "best_trial": best["trial"], "best_val_macro_f1": best["best_val_macro_f1"],
                         "best_val_acc": best["best_val_acc"], "best_val_loss": best["best_val_loss"]}
    arch_order = {a: i for i, a in enumerate(ARCHITECTURES)}
    reference = sorted(by_arch.values(), key=lambda x: (-x["best_val_macro_f1"], -x["best_val_acc"], x["best_val_loss"], arch_order[x["architecture"]]))[0]["architecture"]
    seeds = final_seed_list(args)
    selection = {"created_utc": datetime.now(timezone.utc).isoformat(), "selection_metric": "best tuned validation macro F1",
                 "reference_architecture": reference, "representative_seed": 42 if 42 in seeds else seeds[0],
                 "architectures": args.architectures, "seeds": seeds, "split_sha256": json.loads((out / "split_metadata.json").read_text(encoding="utf-8"))["split_sha256"],
                 "test_metrics_read": False}
    save_json(tuning_dir / "best_params_by_arch.json", by_arch)
    save_json(out / "selection.json", selection)
    save_json(tuning_dir / "sweep_summary.json", sweep_meta)
    save_csv(tuning_dir / "all_trials.csv", all_results)
    return {"selection": selection, "best_params": by_arch, "sweeps": sweep_meta}


def run_final(args, out: Path) -> dict[str, Any]:
    selection_path, params_path = out / "selection.json", out / "tuning" / "best_params_by_arch.json"
    if not selection_path.exists() or not params_path.exists():
        raise FileNotFoundError("Final testing is locked behind tuning artifacts. Run: python pipeline.py --stage tune")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if args.architectures != selection["architectures"] or final_seed_list(args) != selection["seeds"]:
        raise ValueError("Final architecture list/seeds differ from selection.json; refusing test access")
    split = load_split(out / "splits.csv")
    split_meta = json.loads((out / "split_metadata.json").read_text(encoding="utf-8"))
    if split_meta["split_sha256"] != selection["split_sha256"]:
        raise ValueError("Canonical split changed after model selection; refusing test access")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    results = []
    resume_log = []
    split_sha256 = __import__("hashlib").sha256((out / "splits.csv").read_bytes()).hexdigest()
    for arch in selection["architectures"]:
        if arch not in params:
            raise ValueError(f"Missing locked hyperparameters for {arch}")
        config = {**default_config(args), **params[arch], "augmentation": True, "use_class_weights": True}
        for seed in selection["seeds"]:
            resumed = _find_resumable_final_run(out, arch, int(seed), config, split_sha256) if args.resume else None
            if resumed is not None:
                results.append(resumed)
                resume_log.append({"architecture": arch, "seed": seed, "action": "reused_completed_run",
                                   "run_dir": resumed["run_dir"], "checkpoint": resumed["checkpoint"]})
                continue
            try:
                result = fit_run(root=args.data_root, split_csv=out / "splits.csv", out=out, architecture=arch,
                                 seed=seed, config=config, stage="final", wandb_project=args.wandb_project,
                                 wandb_entity=args.wandb_entity, wandb_mode=args.wandb_mode, device=args.device, workers=args.num_workers,
                                 evaluate_test=True)
            except Exception as exc:
                result = {"architecture": arch, "seed": seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            _append_result(out / "final" / "run_metrics.csv", result)
            resume_log.append({"architecture": arch, "seed": seed,
                               "action": "trained" if result.get("status") == "completed" else "failed",
                               "run_dir": result.get("run_dir"), "error": result.get("error")})
    completed = [r for r in results if r.get("status") == "completed"]
    summaries = summarize_runs(results)
    tests = paired_holm_tests(results, selection["reference_architecture"], architecture_order=selection["architectures"])
    save_json(out / "final" / "run_metrics.json", results)
    save_csv(out / "final" / "architecture_summary.csv", summaries)
    save_csv(out / "final" / "statistical_tests.csv", tests)
    per_class_rows = []
    for arch in selection["architectures"]:
        for label in CLASSES:
            per_run = [r.get("per_class", {}).get(label, {}) for r in completed if r["architecture"] == arch]
            if not per_run:
                continue
            for metric in ("precision", "recall", "f1"):
                values = [float(r[metric]) for r in per_run if metric in r]
                per_class_rows.append({"architecture": arch, "class": label, "metric": metric,
                                       "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) >= 2 else None})
            per_class_rows.append({"architecture": arch, "class": label, "metric": "support",
                                   "mean": per_run[0].get("support"), "sd": 0.0})
    save_csv(out / "final" / "per_class_summary.csv", per_class_rows)
    save_json(out / "final" / "resume_decisions.json", resume_log)
    return {"completed_runs": len(completed), "expected_runs": len(selection["architectures"]) * len(selection["seeds"]),
            "selection": selection, "architecture_summary": summaries, "statistical_tests": tests}


def _find_resumable_final_run(out: Path, architecture: str, seed: int,
                              expected: dict[str, Any], split_sha256: str) -> dict[str, Any] | None:
    """Reuse only a complete final result whose saved configuration and split are identical."""
    required = {"optimizer", "learning_rate", "weight_decay", "batch_size", "epochs",
                "input_size", "black_threshold", "augmentation", "use_class_weights"}
    for config_path in sorted((out / "final").glob("*/config.json")):
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if cfg.get("stage") != "final" or cfg.get("architecture") != architecture or int(cfg.get("seed", -1)) != seed:
                continue
            if cfg.get("split_sha256") != split_sha256 or any(cfg.get(key) != expected.get(key) for key in required):
                continue
            run_dir = config_path.parent
            metrics_path, test_path = run_dir / "run_metrics.json", run_dir / "test_metrics.json"
            checkpoint = run_dir / f"best_{architecture}_seed{seed}.pt"
            if not (metrics_path.is_file() and test_path.is_file() and checkpoint.is_file()):
                continue
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            if result.get("status") != "completed" or not all(k in result for k in
                    ("test_acc", "test_macro_precision", "test_macro_recall", "test_macro_f1", "per_class")):
                continue
            result["run_dir"], result["checkpoint"] = str(run_dir), str(checkpoint)
            result["resumed"] = True
            return result
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return None


def run_ablation(args, out: Path) -> list[dict[str, Any]]:
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    arch, seed = selection["reference_architecture"], int(selection["representative_seed"])
    final_runs = json.loads((out / "final" / "run_metrics.json").read_text(encoding="utf-8"))
    baseline = next((r for r in final_runs if r["architecture"] == arch and int(r["seed"]) == seed and r.get("status") == "completed"), None)
    if baseline is None:
        raise FileNotFoundError("A matched full-pipeline final run is required before ablation")
    best_params = json.loads((out / "tuning" / "best_params_by_arch.json").read_text(encoding="utf-8"))[arch]
    base_cfg = {**default_config(args), **best_params, "augmentation": True, "use_class_weights": True}
    variants = [
        ("ablation_no_augmentation", args.data_root, base_cfg | {"augmentation": False}, out / "splits.csv"),
        ("ablation_no_class_weights", args.data_root, base_cfg | {"use_class_weights": False}, out / "splits.csv"),
    ]
    padded_properties = out / "dataset_audit_padding" / "image_properties.csv"
    if padded_properties.exists():
        props = list(csv.DictReader(padded_properties.open(encoding="utf-8-sig", newline="")))
        padding_lookup = {(Path(r["path"]).parent.name, Path(r["path"]).stem): r["path"] for r in props}
        clean_split = load_split(out / "splits.csv")
        padded_split = []
        for row in clean_split:
            key = (Path(row["path"]).parent.name, Path(row["path"]).stem)
            if key not in padding_lookup:
                raise ValueError(f"No paired padded image for {row['image_id']}")
            padded_split.append(row | {"path": padding_lookup[key]})
        pad_csv = out / "ablation" / "paired_padding_split.csv"
        save_csv(pad_csv, padded_split)
        variants.append(("ablation_original_black_padding", args.padded_root, base_cfg, pad_csv))
    results = []
    for stage_name, root, config, split_csv in variants:
        try:
            result = fit_run(root=root, split_csv=split_csv, out=out, architecture=arch, seed=seed,
                             config=config, stage=stage_name, wandb_project=args.wandb_project,
                             wandb_entity=args.wandb_entity,
                             wandb_mode=args.wandb_mode, device=args.device, workers=args.num_workers,
                             evaluate_test=True)
            result["variant"] = stage_name
            result["delta_accuracy_vs_full"] = result["test_acc"] - float(baseline["test_acc"])
            result["delta_macro_f1_vs_full"] = result["test_macro_f1"] - float(baseline["test_macro_f1"])
        except Exception as exc:
            result = {"architecture": arch, "seed": seed, "variant": stage_name, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
    save_json(out / "ablation" / "ablation_results.json", results)
    save_csv(out / "ablation" / "ablation_results.csv", results)
    return results

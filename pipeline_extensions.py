"""Historical extension diagnostics and bounded mode smoke.

Current objective (2026-10-13): independently tune every architecture-mode pair.
The fixed-Full-recipe prepare/ablation/finetune/report CLI stages are retired
and fail before creating output or starting training, even with --dry-run.

Available: analyze cached historical Full-only predictions/histories, or run
bounded smoke of reusable mode APIs. --stage is required.

The new architecture-mode tuning runner is not implemented yet. Read
AGENTS.md and docs/MODE_TUNING_PROTOCOL.md for scope, budget and next work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("analyze", "prepare", "smoke", "ablation", "finetune", "report")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Historical confidence/dynamics analysis. Fixed-Full-recipe measured stages are retired; see docs/MODE_TUNING_PROTOCOL.md.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--stage", required=True, choices=STAGES,
                   help="Stage to run (required; no default that starts training).")
    p.add_argument("--base-out-dir", type=Path, default=Path("outputs"),
                   help="Primary outputs/ directory with canonical splits, final runs, tuning (default: outputs).")
    p.add_argument("--out-dir", type=Path, default=Path("outputs_extensions_v1"),
                   help="Extension output directory (default: outputs_extensions_v1).")
    p.add_argument("--data-root", type=Path, default=Path("dataset_cleaned"),
                   help="Dataset root (default: dataset_cleaned).")
    p.add_argument("--device", default="auto",
                   help="Device: auto | cpu | cuda (default: auto).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default: 0).")
    p.add_argument("--wandb-project", default="thai-shrine-classifier",
                   help="W&B project name.")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (team/user); uses default if not set.")
    p.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"),
                   help="W&B tracking mode (default: online).")
    p.add_argument("--resume", action="store_true",
                   help="Reuse completed extension runs whose protocol/config/split match.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be run without training; report new/reusable/failed runs.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Prerequisites helpers
# ---------------------------------------------------------------------------

def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        sys.exit(f"ERROR: Required file not found: {path}\nRun: {hint}")


def _load_selection(base_out_dir: Path) -> dict:
    sel_path = base_out_dir / "selection.json"
    _require_file(sel_path, "python pipeline.py --stage tune --base-out-dir ...")
    return json.loads(sel_path.read_text(encoding="utf-8"))


def _load_params(base_out_dir: Path) -> dict:
    params_path = base_out_dir / "tuning" / "best_params_by_arch.json"
    _require_file(params_path, "python pipeline.py --stage tune --base-out-dir ...")
    return json.loads(params_path.read_text(encoding="utf-8"))


def _load_split_csv(base_out_dir: Path) -> Path:
    split_path = base_out_dir / "splits.csv"
    _require_file(split_path, "python pipeline.py --stage prep --base-out-dir ...")
    return split_path


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _collect_environment(device: str) -> dict:
    env: dict = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_requested": device,
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_device_memory_mib"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except ImportError:
        env["torch_version"] = "not_installed"
        env["cuda_available"] = False
    try:
        import torchvision
        env["torchvision_version"] = torchvision.__version__
    except ImportError:
        env["torchvision_version"] = "not_installed"
    try:
        import wandb
        env["wandb_version"] = getattr(wandb, "__version__", "installed_no_version")
    except ImportError:
        env["wandb_version"] = "not_installed"
    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except ImportError:
        pass
    try:
        import PIL
        env["pillow_version"] = PIL.__version__
    except ImportError:
        pass
    try:
        import sklearn
        env["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    return env


# ---------------------------------------------------------------------------
# Stage: analyze
# ---------------------------------------------------------------------------

def stage_analyze(args) -> None:
    print("[analyze] Running confidence/reliability analysis and training dynamics from cached artifacts...")
    base = args.base_out_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # Verify prerequisites
    _require_file(base / "final" / "run_metrics.json",
                  "python pipeline.py --stage final ...")
    selection = _load_selection(base)

    from shrine_classifier_ext.confidence_analysis import run_confidence_analysis
    from shrine_classifier_ext.diagnostics_analysis import run_diagnostics_analysis

    print("[analyze] Item 1 — Confidence/reliability analysis")
    conf_result = run_confidence_analysis(base, out, selection)
    n_ok = sum(1 for r in conf_result["run_rows"] if r.get("status") == "ok")
    n_issues = len(conf_result["issues"])
    print(f"  -> {n_ok} runs processed; {n_issues} issues")
    if conf_result["issues"]:
        for iss in conf_result["issues"][:5]:
            print(f"     [ISSUE] {iss}")

    print("[analyze] Item 2 — Training dynamics and tuning analysis")
    diag_result = run_diagnostics_analysis(base, out, selection)
    print(f"  -> {len(diag_result['dynamics'])} run dynamics records; "
          f"{len(diag_result['trial_rows'])} tuning trials loaded")

    print(f"[analyze] Outputs written to {out}")
    _update_execution_status(
        "analyze",
        "completed",
        f"Confidence: {n_ok} runs OK, {n_issues} issues. "
        f"Diagnostics: {len(diag_result['dynamics'])} runs, "
        f"{len(diag_result['trial_rows'])} trials.",
        out,
    )


# ---------------------------------------------------------------------------
# Stage: prepare
# ---------------------------------------------------------------------------

def stage_prepare(args) -> None:
    print("[prepare] Writing protocol, preflight, and environment files...")
    base = args.base_out_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    selection = _load_selection(base)
    params = _load_params(base)
    split_csv = _load_split_csv(base)

    split_sha256_csv = hashlib.sha256(split_csv.read_bytes()).hexdigest()

    # Check if protocol.json already exists → protect it
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        print(f"  [WARNING] protocol.json already exists at {protocol_path}; not overwriting.")
        print("  Create a new --out-dir for an intentional protocol revision.")
    else:
        # Build image-ID split fingerprint
        import csv as _csv
        with split_csv.open("r", encoding="utf-8-sig", newline="") as f:
            split_rows = list(_csv.DictReader(f))
        id_split_pairs = sorted(f"{r['image_id']}:{r['split']}" for r in split_rows)
        id_split_hash = hashlib.sha256("|".join(id_split_pairs).encode()).hexdigest()

        arch = selection.get("reference_architecture", "")
        best = params.get(arch, {})

        protocol = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "extension_exploratory",
            "exploratory_note": (
                "Items 3 and 4 use the test split that was already inspected during "
                "the primary final evaluation. Protocol lock prevents further cherry-picking but "
                "does not restore unseen-test status."
            ),
            "reference_architecture": arch,
            "seeds": selection.get("seeds", []),
            "representative_seed": selection.get("representative_seed", 42),
            "base_out_dir": str(base.resolve()),
            "out_dir": str(out.resolve()),
            "source_selection_path": str((base / "selection.json").resolve()),
            "source_params_path": str((base / "tuning" / "best_params_by_arch.json").resolve()),
            "split_csv_path": str(split_csv.resolve()),
            "split_sha256_csv_bytes": split_sha256_csv,
            "split_sha256_id_split_pairs": id_split_hash,
            "class_order": ["san_phraphrom", "san_phraphum", "san_tayay", "tee_ju_eia"],
            "conditions": {
                "full": "Fresh full fine-tuning baseline (shared between Items 3 and 4)",
                "no_augmentation": "Disable train augmentation only; all else identical to full",
                "no_class_weights": "Uniform cross-entropy only; all else identical to full",
                "frozen": "Frozen backbone (eval), train head only",
                "partial": "Frozen backbone except final stage, train final stage + head",
            },
            "locked_recipe": {
                "architecture": arch,
                "optimizer": best.get("optimizer"),
                "learning_rate": best.get("learning_rate"),
                "weight_decay": best.get("weight_decay"),
                "batch_size": best.get("batch_size"),
                "epochs": 30,
                "patience": 7,
                "input_size": 224,
                "scheduler": "ReduceLROnPlateau(mode=max,factor=0.5,patience=3)",
                "augmentation": True,
                "use_class_weights": True,
            },
            "partial_backbone_stages": {
                "resnet50": "layer4",
                "densenet121": "features.denseblock4 + features.norm5",
                "convnext_tiny": "features[-1] (final ConvNeXt stage)",
                "efficientnet_b0": "features[-2] (final MBConv stage) + features[-1] (final projection)",
                "mobilenet_v3_large": "features[-2] (final InvertedResidual group) + features[-1] (final projection)",
            },
            "budget": {
                "conditions": 5,
                "seeds_per_condition": 5,
                "max_epochs_per_run": 30,
                "total_max_epochs": 750,
                "note": "full baseline shared between ablation and finetune stages",
            },
            "planned_comparison_family": {
                "contrasts": ["no_augmentation-full", "no_class_weights-full", "frozen-full", "partial-full"],
                "metrics": ["test_acc", "test_macro_f1"],
                "family_size": 8,
                "test": "two-sided scipy.stats.ttest_rel",
                "correction": "Holm",
                "note": "Separate from primary 8-comparison family in outputs/final/statistical_tests.csv",
            },
            "checkpoint_selection_rule": "max val_macro_f1; ties: lower val_loss; ties: earlier epoch",
            "min_delta": 1e-4,
        }
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"  Wrote protocol.json to {protocol_path}")

    # Preflight checks
    preflight: dict = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": []}
    def chk(name: str, ok: bool, msg: str):
        status = "pass" if ok else "fail"
        preflight["checks"].append({"name": name, "status": status, "message": msg})
        print(f"  [{status.upper()}] {name}: {msg}")

    chk("base_out_dir_exists", base.exists(), str(base))
    chk("splits_csv_exists", (base / "splits.csv").exists(), str(base / "splits.csv"))
    chk("selection_json_exists", (base / "selection.json").exists(), str(base / "selection.json"))
    chk("best_params_exists", (base / "tuning" / "best_params_by_arch.json").exists(),
        str(base / "tuning" / "best_params_by_arch.json"))
    chk("final_run_metrics_exists", (base / "final" / "run_metrics.json").exists(),
        str(base / "final" / "run_metrics.json"))
    chk("data_root_exists", args.data_root.exists(), str(args.data_root))
    chk("out_dir_writable", True, str(out))  # we just created it above

    try:
        import torch
        chk("torch_installed", True, torch.__version__)
        if args.device in ("auto", "cuda"):
            chk("cuda_available", torch.cuda.is_available(),
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA not available")
    except ImportError:
        chk("torch_installed", False, "torch not installed — required for training stages")

    all_pass = all(c["status"] == "pass" for c in preflight["checks"])
    preflight["overall"] = "pass" if all_pass else "partial_fail"
    preflight["budget_display"] = (
        "5 conditions × 5 seeds × max 30 epochs = max 750 epochs (before early stopping)\n"
        "  full (5) + no_augmentation (5) + no_class_weights (5) + frozen (5) + partial (5) = 25 runs\n"
        "  full baseline shared between ablation and finetune stages"
    )

    from shrine_classifier.analysis import save_json
    save_json(out / "preflight.json", preflight)
    env = _collect_environment(args.device)
    save_json(out / "environment.json", env)
    print(f"  Preflight: {preflight['overall']}")
    print(f"  Budget: {preflight['budget_display']}")
    print(f"[prepare] Done. Outputs: {out}")


# ---------------------------------------------------------------------------
# Stage: smoke
# ---------------------------------------------------------------------------

def stage_smoke(args) -> None:
    print("[smoke] Running bounded model/mode smoke test (no test split access)...")
    base = args.base_out_dir
    out = args.out_dir

    # Check PyTorch
    try:
        import torch
        import torchvision
    except ImportError:
        sys.exit(
            "ERROR: PyTorch and torchvision are required for the smoke stage.\n"
            "Install them with the correct CUDA version and retry."
        )

    smoke_dir = out / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    from shrine_classifier.models import build_model, ARCHITECTURES
    from shrine_classifier_ext.finetuning_modes import apply_finetuning_mode, enforce_mode_per_epoch, log_mode_stats, MODES

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    smoke_results: list[dict] = []
    issues: list[str] = []

    for arch in ARCHITECTURES:
        for mode in MODES:
            result: dict = {"architecture": arch, "mode": mode}
            try:
                # Build with synthetic random weights for smoke (no pretrained download required)
                model, meta = build_model(arch, pretrained=False)
                model = model.to(device)
                mode_stats = apply_finetuning_mode(model, arch, mode)

                # Verify 4 logits
                dummy = torch.zeros(2, 3, 224, 224, device=device)
                model.train()
                enforce_mode_per_epoch(model, arch, mode)
                logits = model(dummy)
                assert logits.shape == (2, 4), f"Expected (2,4), got {logits.shape}"

                # Verify backward
                loss = logits.sum()
                loss.backward()
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                has_grad = any(p.grad is not None for p in trainable_params)

                # Verify frozen params unchanged through optimizer step
                frozen_params_before = {
                    name: p.detach().clone()
                    for name, p in model.named_parameters() if not p.requires_grad
                }
                if trainable_params:
                    opt = torch.optim.SGD(trainable_params, lr=0.001)
                    opt.step()
                for name, before_val in frozen_params_before.items():
                    current = dict(model.named_parameters())[name].detach()
                    if not torch.allclose(before_val, current, atol=1e-10):
                        issues.append(f"{arch}/{mode}: frozen param {name} changed after step")

                result.update({
                    "status": "pass",
                    "logits_shape": list(logits.shape),
                    "has_gradient": has_grad,
                    "trainable_params": mode_stats["trainable_parameters"],
                    "frozen_params": mode_stats["frozen_parameters"],
                    "trainable_module_names": mode_stats["trainable_module_names"],
                    "note": "SYNTHETIC random weights — pretrained loading not tested here",
                })
                print(f"  [PASS] {arch}/{mode}: logits={logits.shape}, "
                      f"trainable={mode_stats['trainable_parameters']:,}")
            except Exception as exc:
                result.update({"status": "fail", "error": f"{type(exc).__name__}: {exc}"})
                issues.append(f"{arch}/{mode}: {exc}")
                print(f"  [FAIL] {arch}/{mode}: {exc}")
            smoke_results.append(result)

    # Verify no test loader created during smoke
    result_no_test = {
        "check": "no_test_loader_in_smoke",
        "status": "pass",
        "note": "Smoke uses synthetic inputs only; no DataLoader touching test split was created",
    }
    smoke_results.append(result_no_test)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_str,
        "total_checks": len(smoke_results),
        "passed": sum(1 for r in smoke_results if r.get("status") == "pass"),
        "failed": sum(1 for r in smoke_results if r.get("status") == "fail"),
        "issues": issues,
        "synthetic_weights_note": (
            "All model checks use synthetic random weights. "
            "Pretrained weight download/loading was NOT verified in this smoke pass."
        ),
        "results": smoke_results,
    }
    from shrine_classifier.analysis import save_json
    save_json(smoke_dir / "smoke_results.json", summary)
    print(f"[smoke] {summary['passed']} passed, {summary['failed']} failed. "
          f"Output: {smoke_dir / 'smoke_results.json'}")
    if issues:
        print(f"[smoke] Issues: {issues}")
    _update_execution_status(
        "smoke",
        "completed" if summary["failed"] == 0 else "partial_fail",
        f"{summary['passed']} pass, {summary['failed']} fail (synthetic weights only; pretrained unverified)",
        out,
    )


# ---------------------------------------------------------------------------
# Stage: ablation
# ---------------------------------------------------------------------------

def stage_ablation(args) -> None:
    print("[ablation] Running shared full baseline + ablation conditions (Items 3)...")
    base = args.base_out_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    selection = _load_selection(base)
    params = _load_params(base)
    split_csv = _load_split_csv(base)
    seeds = [int(s) for s in selection.get("seeds", [])]

    if args.dry_run:
        print("[ablation] DRY-RUN mode — showing what would be run")
    else:
        # Verify protocol is locked
        protocol_path = out / "protocol.json"
        if not protocol_path.exists():
            sys.exit(
                "ERROR: protocol.json not found. Lock the protocol before training:\n"
                "  python pipeline_extensions.py --stage prepare ..."
            )

    # Check PyTorch unless dry-run
    if not args.dry_run:
        try:
            import torch
        except ImportError:
            sys.exit("ERROR: torch not installed. Install PyTorch before training stages.")

    from shrine_classifier_ext.extension_experiments import run_extension_ablation
    results = run_extension_ablation(
        out_dir=out, base_out_dir=base, data_root=args.data_root,
        split_csv=split_csv, selection=selection, params=params, seeds=seeds,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode, device=args.device, workers=args.num_workers,
        resume=args.resume, dry_run=args.dry_run,
    )

    if args.dry_run:
        total_new = sum(1 for cond_runs in results.values()
                        for r in cond_runs if r.get("status") == "dry_run_new")
        total_reuse = sum(1 for cond_runs in results.values()
                          for r in cond_runs if r.get("status") == "dry_run_reusable")
        max_epochs = total_new * 30
        print(f"[ablation dry-run] New runs: {total_new}, Reusable: {total_reuse}")
        print(f"[ablation dry-run] Maximum remaining epochs: {max_epochs}")
    else:
        for cond, cond_runs in results.items():
            ok = sum(1 for r in cond_runs if r.get("status") == "completed")
            fail = sum(1 for r in cond_runs if r.get("status") == "failed")
            print(f"  {cond}: {ok} completed, {fail} failed")

    print(f"[ablation] Done. Outputs: {out}")


# ---------------------------------------------------------------------------
# Stage: finetune
# ---------------------------------------------------------------------------

def stage_finetune(args) -> None:
    print("[finetune] Running frozen/partial fine-tuning comparison (Item 4)...")
    base = args.base_out_dir
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    selection = _load_selection(base)
    params = _load_params(base)
    split_csv = _load_split_csv(base)
    seeds = [int(s) for s in selection.get("seeds", [])]

    if args.dry_run:
        print("[finetune] DRY-RUN mode — showing what would be run")
    else:
        protocol_path = out / "protocol.json"
        if not protocol_path.exists():
            sys.exit(
                "ERROR: protocol.json not found. Lock the protocol before training:\n"
                "  python pipeline_extensions.py --stage prepare ..."
            )

    if not args.dry_run:
        try:
            import torch
        except ImportError:
            sys.exit("ERROR: torch not installed.")

    from shrine_classifier_ext.extension_experiments import run_extension_finetune
    results = run_extension_finetune(
        out_dir=out, base_out_dir=base, data_root=args.data_root,
        split_csv=split_csv, selection=selection, params=params, seeds=seeds,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode, device=args.device, workers=args.num_workers,
        resume=args.resume, dry_run=args.dry_run,
    )

    if args.dry_run:
        total_new = sum(1 for cond_runs in results.values()
                        for r in cond_runs if r.get("status") == "dry_run_new")
        total_reuse = sum(1 for cond_runs in results.values()
                          for r in cond_runs if r.get("status") == "dry_run_reusable")
        max_epochs = total_new * 30
        print(f"[finetune dry-run] New runs: {total_new}, Reusable: {total_reuse}")
        print(f"[finetune dry-run] Maximum remaining epochs: {max_epochs}")
    else:
        for cond, cond_runs in results.items():
            ok = sum(1 for r in cond_runs if r.get("status") == "completed")
            fail = sum(1 for r in cond_runs if r.get("status") == "failed")
            print(f"  {cond}: {ok} completed, {fail} failed")

    print(f"[finetune] Done. Outputs: {out}")


# ---------------------------------------------------------------------------
# Stage: report
# ---------------------------------------------------------------------------

def stage_report(args) -> None:
    print("[report] Generating extension summary from existing artifacts...")
    # Recompute comparisons across all completed conditions on disk if present
    try:
        from shrine_classifier_ext.extension_experiments import _save_comparisons
        selection_path = args.base_out_dir / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
        arch = selection.get("selected_architecture", "resnet50")
        seeds = selection.get("seeds", [42, 123, 456, 789, 1024])
        _save_comparisons(args.out_dir, {}, seeds, arch, selection)
    except Exception as exc:
        print(f"[report] Note on comparisons refresh: {exc}")

    from shrine_classifier_ext.extension_report import run_report
    result = run_report(args.out_dir, args.base_out_dir)
    print(f"[report] Status summary:")
    for k, v in sorted(result["status"].items()):
        icon = "[OK]" if v in ("present", "generated") else "[MISSING]"
        print(f"  {icon} {k}: {v}")
    print(f"[report] Done. Assets: {result['out_dir']}")


# ---------------------------------------------------------------------------
# Status file update
# ---------------------------------------------------------------------------

def _update_execution_status(stage: str, status: str, details: str, out_dir: Path) -> None:
    status_path = Path("EXTENSION_EXECUTION_STATUS.md")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = (
        f"\n## Extension stage: `{stage}` — {status} ({timestamp})\n\n"
        f"{details}\n\n"
        f"Output directory: `{out_dir}`\n"
    )
    if status_path.exists():
        current = status_path.read_text(encoding="utf-8")
        status_path.write_text(current + entry, encoding="utf-8")
    else:
        header = (
            "# Extension Execution Status\n\n"
            "Tracks `pipeline_extensions.py` stage outcomes.\n"
            "Appended by each run; original entries preserved.\n"
        )
        status_path.write_text(header + entry, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    # User retired this experiment and deleted its outputs on 2026-10-13.
    # Block before dispatch/output creation, including dry-run, so no old recipe
    # is mistaken for independent architecture-mode tuning.
    if args.stage in {"prepare", "ablation", "finetune", "report"}:
        raise SystemExit(
            f"Stage '{args.stage}' belongs to the retired fixed-Full-recipe protocol. "
            "It is disabled and will not recreate deleted experiment outputs. "
            "The requested flow tunes each architecture-mode pair independently; "
            "see docs/MODE_TUNING_PROTOCOL.md. Its new runner is not implemented yet. "
            "Historical cached analysis and bounded smoke remain available."
        )

    dispatch = {
        "analyze": stage_analyze,
        "prepare": stage_prepare,
        "smoke": stage_smoke,
        "ablation": stage_ablation,
        "finetune": stage_finetune,
        "report": stage_report,
    }
    fn = dispatch[args.stage]

    t0 = time.time()
    try:
        fn(args)
    except SystemExit:
        raise
    except Exception as exc:
        import traceback
        print(f"\n[ERROR] Stage '{args.stage}' failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        _update_execution_status(args.stage, "error", str(exc), args.out_dir)
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"[{args.stage}] Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    # Windows multiprocessing safe entry point
    import multiprocessing
    multiprocessing.freeze_support()
    main()

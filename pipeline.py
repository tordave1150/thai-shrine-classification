from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

from shrine_classifier.data import CLASSES, audit_dataset, create_split


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Thai shrine image classification experiment pipeline")
    p.add_argument("--stage", required=True, choices=["audit", "prep", "baseline", "screen", "tune", "final", "gradcam", "eyeball", "report", "ablation", "smoke", "all"])
    p.add_argument("--data-root", type=Path, default=Path("dataset_cleaned"))
    p.add_argument("--padded-root", type=Path, default=Path("dataset_padding"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--runs", type=int)
    p.add_argument("--trials", type=int, default=12, help="Number of trials per architecture")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--wandb-project", default="tordave1150-thai-shrine-v1")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seeds", help="Comma-separated explicit final seed list")
    p.add_argument("--architectures", nargs="+", choices=["resnet50", "efficientnet_b0", "mobilenet_v3_large", "densenet121", "convnext_tiny"], default=["resnet50", "efficientnet_b0", "mobilenet_v3_large", "densenet121", "convnext_tiny"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--black-threshold", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--near-threshold", type=int, default=6)
    p.add_argument("--duplicate-review", type=Path, help="CSV with id_a,id_b,review_status columns")
    return p


def require_torchvision():
    try:
        import torch
        import torchvision
        return torch, torchvision
    except Exception as exc:
        raise RuntimeError("Model/training stages need compatible PyTorch and torchvision. Install the runtime-specific official PyTorch build; no random-weight fallback is used.") from exc


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    out = args.out_dir.resolve()
    stage = args.stage
    if stage == "smoke" and args.out_dir == Path("outputs"):
        out = (Path.cwd() / "outputs_smoke").resolve()
    if args.dry_run:
        count = len(args.architectures)
        runs = args.runs if args.runs is not None else 5
        print(json.dumps({"stage": stage, "data_root": str(args.data_root.resolve()), "out_dir": str(out),
                          "architectures": args.architectures,
                          "estimated_training_runs": count * (1 + args.trials + runs) if stage == "all" else None,
                          "max_epochs": count * (1 + args.trials + runs) * args.epochs if stage == "all" else None,
                          "test_metrics_access": stage in {"final", "gradcam", "eyeball", "report", "all"}}, indent=2))
        return 0
    if stage in {"audit", "all"}:
        summary = audit_dataset(args.data_root, out / "dataset_audit", near_threshold=args.near_threshold)
        padded_summary = audit_dataset(args.padded_root, out / "dataset_audit_padding", near_threshold=args.near_threshold, require_metadata=False)
        cleaned_ids = {(Path(r["path"]).parent.name, Path(r["path"]).stem) for r in _read_properties(out / "dataset_audit" / "image_properties.csv")}
        padded_ids = {(Path(r["path"]).parent.name, Path(r["path"]).stem) for r in _read_properties(out / "dataset_audit_padding" / "image_properties.csv")}
        pair_report = {"cleaned_count": len(cleaned_ids), "padding_count": len(padded_ids),
                       "paired_by_class_and_stem": len(cleaned_ids & padded_ids),
                       "cleaned_only": sorted("/".join(k) for k in cleaned_ids - padded_ids),
                       "padding_only": sorted("/".join(k) for k in padded_ids - cleaned_ids)}
        (out / "dataset_audit" / "version_pairing.json").write_text(json.dumps(pair_report, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_environment(out / "environment.json")
        print(json.dumps({"stage": "audit", "num_images": summary["num_images"], "class_counts": summary["class_counts"],
                          "duplicate_groups": summary["exact_decoded_pixel_duplicate_groups"],
                          "near_duplicate_candidates": summary["num_near_duplicate_candidates"],
                          "padding_images": padded_summary["num_images"], "version_pairs": pair_report["paired_by_class_and_stem"],
                          "errors": len(summary["errors"]) + len(padded_summary["errors"]), "outputs": str(out / "dataset_audit")}, ensure_ascii=False, indent=2))
        if summary["errors"] or padded_summary["errors"] or pair_report["cleaned_only"] or pair_report["padding_only"]:
            return 2
        if stage == "audit":
            return 0
    if stage == "prep":
        audit = out / "dataset_audit"
        if not (audit / "image_properties.csv").exists():
            raise RuntimeError("Missing audit artifacts. Run: python pipeline.py --stage audit")
        existing = out / "splits.csv"
        if existing.exists():
            _check_existing_split(out, args.data_root)
            metadata = json.loads((out / "split_metadata.json").read_text(encoding="utf-8"))
            if metadata.get("split_seed") != args.split_seed:
                raise RuntimeError("Existing canonical split uses a different seed; use a new --out-dir")
            print(json.dumps({"stage": "prep", "status": "reused_existing_split", "split": metadata}, indent=2))
            return 0
        saved_review = audit / "near_duplicate_review_decisions.csv"
        review = args.duplicate_review or (saved_review if saved_review.exists() else None)
        split = create_split(args.data_root, audit, existing, split_seed=args.split_seed, review_csv=review)
        print(json.dumps({"stage": "prep", "split": split}, ensure_ascii=False, indent=2))
        return 0
    if stage == "smoke":
        return _run_smoke(args, out)
    if stage == "all" and not (out / "splits.csv").exists():
        review = out / "dataset_audit" / "near_duplicate_review_decisions.csv"
        create_split(args.data_root, out / "dataset_audit", out / "splits.csv",
                     split_seed=args.split_seed, review_csv=review if review.exists() else args.duplicate_review)
    if stage == "all" and (out / "splits.csv").exists():
        _check_existing_split(out, args.data_root)
    if stage in {"baseline", "screen", "tune", "final", "gradcam", "eyeball", "report", "ablation", "all"}:
        if stage == "report":
            results = _make_report(out)
            print(json.dumps({"stage": stage, "outputs": str(out), "results": results}, ensure_ascii=False, indent=2))
            return 0
        if stage != "report":
            require_torchvision()
        from shrine_classifier.experiments import run_ablation, run_final, run_screen, run_tuning
        results = {}
        stages = ["baseline", "screen", "tune", "final", "gradcam", "eyeball", "report"] if stage == "all" else [stage]
        for item in stages:
            if item in {"baseline", "screen", "tune", "final", "ablation"}:
                if item != "baseline" and not (out / "splits.csv").exists():
                    raise FileNotFoundError("Missing canonical split. Run: python pipeline.py --stage audit, then --stage prep")
            if item == "baseline":
                from shrine_classifier.baseline import run_baseline
                results[item] = run_baseline(args, out)
            elif item == "screen":
                results[item] = run_screen(args, out)
            elif item == "tune":
                results[item] = run_tuning(args, out)
            elif item == "final":
                results[item] = run_final(args, out)
            elif item == "ablation":
                results[item] = run_ablation(args, out)
            elif item in {"gradcam", "eyeball"}:
                from shrine_classifier.visualization import make_analysis_grids
                results[item] = make_analysis_grids(args, out, mode=item)
            elif item == "report":
                results[item] = _make_report(out)
        print(json.dumps({"stage": stage, "outputs": str(out), "results": results}, ensure_ascii=False, indent=2, default=str))
        return 0
    return 0


def _read_properties(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _check_existing_split(out: Path, data_root: Path) -> None:
    with (out / "splits.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        split = list(csv.DictReader(stream))
    if not split:
        raise RuntimeError("Existing splits.csv is empty")
    required = {"image_id", "path", "label", "label_index", "group_id", "split"}
    if not required.issubset(split[0]):
        raise RuntimeError("Existing split schema is invalid; use a new output directory")
    group_splits = {}
    for row in split:
        if row["label"] not in CLASSES or int(row["label_index"]) != CLASSES.index(row["label"]):
            raise RuntimeError("Existing split class order/index mismatch")
        previous = group_splits.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise RuntimeError("Existing duplicate group crosses splits")
    meta_path = out / "split_metadata.json"
    if not meta_path.exists():
        raise RuntimeError("Existing splits.csv lacks split_metadata.json; use a new output directory")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = __import__("hashlib").sha256("\n".join(f"{r['image_id']}:{r['split']}" for r in sorted(split, key=lambda x: x["image_id"])).encode()).hexdigest()
    if digest != meta.get("split_sha256"):
        raise RuntimeError("Existing split fingerprint mismatch; refusing to continue or regenerate")
    props_path = out / "dataset_audit" / "image_properties.csv"
    from shrine_classifier.data import groups_from_audit
    with props_path.open("r", encoding="utf-8-sig", newline="") as stream:
        props = list(csv.DictReader(stream))
    manifest = __import__("hashlib").sha256("\n".join(f"{r['image_id']}:{r['pixel_sha256']}" for r in props).encode()).hexdigest()
    review = out / "dataset_audit" / "near_duplicate_review_decisions.csv"
    groups = groups_from_audit(props, review if review.exists() else None)
    group_digest = __import__("hashlib").sha256("\n".join(f"{k}:{groups[k]}" for k in sorted(groups)).encode()).hexdigest()
    if manifest != meta.get("manifest_sha256") or group_digest != meta.get("group_sha256"):
        raise RuntimeError("Image manifest or duplicate review changed after splitting; refusing to continue. Use a new --out-dir")
    for row in props:
        current = data_root / row["path"]
        if not current.is_file() or __import__("hashlib").sha256(current.read_bytes()).hexdigest() != row["byte_sha256"]:
            raise RuntimeError(f"Image changed after the audit: {row['path']}; re-audit and use a new --out-dir")
    audit_summary = json.loads((out / "dataset_audit" / "dataset_summary.json").read_text(encoding="utf-8"))
    metadata_file = data_root / "metadata_cleaned.csv"
    if not metadata_file.exists() or __import__("hashlib").sha256(metadata_file.read_bytes()).hexdigest() != audit_summary.get("metadata", {}).get("sha256"):
        raise RuntimeError("Source metadata changed or was not audited; re-audit and use a new --out-dir")
    if review.exists() and meta.get("duplicate_review_sha256"):
        if __import__("hashlib").sha256(review.read_bytes()).hexdigest() != meta["duplicate_review_sha256"]:
            raise RuntimeError("Near-duplicate review changed after splitting; refusing to continue")


def _run_smoke(args, out: Path) -> int:
    from importlib.util import find_spec
    out.mkdir(parents=True, exist_ok=True)
    missing = [name for name in ("torch", "torchvision") if find_spec(name) is None]
    record = {"stage": "smoke", "python": platform.python_version(), "python_executable": sys.executable,
              "torch_installed": "torch" not in missing, "torchvision_installed": "torchvision" not in missing,
              "wandb_installed": find_spec("wandb") is not None,
              "pretrained_weights_tested": False, "synthetic_backward_smoke": "not_run",
              "test_split_accessed": False, "missing_dependencies": missing}
    record["gpu"] = _gpu_inventory()
    try:
        import numpy
        record["numpy"] = numpy.__version__
        import PIL
        record["pillow"] = PIL.__version__
    except Exception as exc:
        record["environment_probe_error"] = f"{type(exc).__name__}: {exc}"
    if missing:
        record["status"] = "blocked_before_model_smoke"
        (out / "smoke_environment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise RuntimeError(f"Smoke could not exercise the five CNNs; missing {', '.join(missing)}. GPU availability is not a substitute for the PyTorch runtime. See {out / 'smoke_environment.json'}")
    import torch
    import torchvision
    record.update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                   "cuda_available": torch.cuda.is_available(),
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    record["status"] = "model_smoke_not_yet_run"
    from shrine_classifier.models import ARCHITECTURES, build_model
    from shrine_classifier.data import IMAGENET_MEAN, IMAGENET_STD
    import numpy as np
    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    dummy = torch.randn(2, 3, args.input_size, args.input_size, device=device)
    results = []
    for arch in ARCHITECTURES:
        entry = {"architecture": arch, "weights": "synthetic random initialization; not a result"}
        try:
            model, metadata = build_model(arch, pretrained=False)
            model.to(device).train()
            logits = model(dummy)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1], device=device))
            loss.backward()
            entry.update({"status": "forward_backward_passed", "logit_shape": list(logits.shape), "loss": float(loss.detach().cpu()), **metadata})
            if arch == "resnet50":
                model.eval()
                cam = __import__("shrine_classifier.analysis", fromlist=["gradcam"]).gradcam(model, arch, dummy[:1].detach(), 0)
                entry["gradcam_shape"] = list(cam.shape)
                entry["gradcam_finite"] = bool(torch.isfinite(cam).all().item())
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as exc:
            entry.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        results.append(entry)
    record["models"] = results
    record["synthetic_backward_smoke"] = "passed" if all(r["status"] == "forward_backward_passed" for r in results) else "failed"
    record["status"] = "completed" if record["synthetic_backward_smoke"] == "passed" else "failed"
    (out / "smoke_environment.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0 if record["status"] == "completed" else 1


def _make_report(out: Path) -> dict:
    assets = out / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for rel in ("dataset_audit/class_distribution.png", "dataset_audit/sample_grid.png",
                "dataset_audit/preprocessing_examples.png", "baseline/pretrained_examples.png",
                "final/architecture_summary.csv", "final/statistical_tests.csv", "gradcam/gradcam_grid.png",
                "eyeball/eyeball_grid.png"):
        (missing if not (out / rel).exists() else rows).append(rel)
    body = "# Thai shrine classifier presentation assets\n\n"
    body += "Available assets:\n" + ("\n".join(f"- `{x}`" for x in rows) if rows else "- None yet") + "\n\n"
    body += "Not yet available:\n" + ("\n".join(f"- `{x}`" for x in missing) if missing else "- None") + "\n"
    (assets / "README.md").write_text(body, encoding="utf-8")
    return {"available": rows, "missing": missing, "readme": str(assets / "README.md")}


def _gpu_inventory() -> dict:
    import subprocess
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5, check=False)
        return {"detected": result.returncode == 0, "inventory": result.stdout.strip(), "error": result.stderr.strip() if result.returncode else None}
    except Exception as exc:
        return {"detected": False, "error": f"{type(exc).__name__}: {exc}"}


def _write_environment(path: Path) -> None:
    import importlib.metadata as metadata
    from importlib.util import find_spec
    packages = {}
    for name in ("numpy", "Pillow", "scipy", "scikit-learn", "matplotlib", "torch", "torchvision", "wandb"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    record = {"timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
              "python": platform.python_version(), "python_executable": sys.executable,
              "packages": packages, "gpu": _gpu_inventory(),
              "torch_importable": bool(find_spec("torch")), "wandb_importable": bool(find_spec("wandb"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .analysis import gradcam, save_csv, save_json
from .baseline import imagenet_predict
from .data import CLASSES, letterbox, open_rgb
from .models import build_model
from .training import ImageDataset, load_split


def make_analysis_grids(args, out: Path, *, mode: str = "both") -> dict:
    selection_path = out / "selection.json"
    final_path = out / "final" / "run_metrics.json"
    if not selection_path.exists() or not final_path.exists():
        raise FileNotFoundError("Run tuning and final evaluation before analysis; use --stage tune, then --stage final")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    run_rows = json.loads(final_path.read_text(encoding="utf-8"))
    arch, seed = selection["reference_architecture"], int(selection["representative_seed"])
    run = next((r for r in run_rows if r["architecture"] == arch and int(r["seed"]) == seed and r.get("status") == "completed"), None)
    if not run:
        raise FileNotFoundError(f"No completed selected checkpoint for {arch}, seed {seed}")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(arch, pretrained=False)
    state = torch.load(run["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"]); model.to(device).eval()
    rows = [r for r in load_split(out / "splits.csv") if r["split"] == "test"]
    dataset = ImageDataset(args.data_root, rows, training=False, size=args.input_size)
    predictions = list(csv.DictReader(Path(run["run_dir"], "test_predictions.csv").open(encoding="utf-8-sig", newline="")))
    by_id = {r["image_id"]: r for r in predictions}
    selected = []
    for cls in CLASSES:
        class_rows = [r for r in rows if r["label"] == cls and r["image_id"] in by_id]
        correct = sorted((r for r in class_rows if by_id[r["image_id"]]["true_label"] == by_id[r["image_id"]]["predicted_label"]),
                         key=lambda r: float(by_id[r["image_id"]]["confidence"]))
        wrong = sorted((r for r in class_rows if by_id[r["image_id"]]["true_label"] != by_id[r["image_id"]]["predicted_label"]),
                       key=lambda r: -float(by_id[r["image_id"]]["confidence"]))
        if wrong:
            selected.append((wrong[0], "misclassified_test_example"))
        if correct:
            selected.append((correct[0], "lowest_confidence_correct_test_example"))
        elif class_rows:
            selected.append((min(class_rows, key=lambda r: float(by_id[r["image_id"]]["confidence"])), "lowest_confidence_available_test_example"))
    visdir = out / "gradcam"; visdir.mkdir(parents=True, exist_ok=True)
    records = []
    if mode in {"both", "gradcam"}:
        for row, reason in selected:
            idx = next(i for i, record in enumerate(rows) if record["image_id"] == row["image_id"])
            tensor, label_index, image_id, bbox = dataset[idx]
            logits = model(tensor.unsqueeze(0).to(device))
            pred_idx = int(logits.argmax(1).item())
            conf = float(logits.softmax(1)[0, pred_idx].item())
            cam = gradcam(model, arch, tensor.unsqueeze(0).to(device), pred_idx)[0].detach().cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            cam_crop = cam[y1:y2, x1:x2]
            original = open_rgb(args.data_root / row["path"])
            heat = _heat_image(cam_crop, original.size)
            overlay = Image.blend(original.convert("RGBA"), heat.convert("RGBA"), .42).convert("RGB")
            stem = row["image_id"].replace("/", "__")
            original.save(visdir / f"{stem}_original.jpg")
            heat.save(visdir / f"{stem}_heatmap.png")
            overlay.save(visdir / f"{stem}_overlay.jpg")
            records.append({"image_id": image_id, "true_label": CLASSES[label_index], "predicted_label": CLASSES[pred_idx],
                            "target_class": CLASSES[pred_idx], "confidence": conf, "selection_reason": reason,
                            "original": f"{stem}_original.jpg", "heatmap": f"{stem}_heatmap.png", "overlay": f"{stem}_overlay.jpg"})
        _cam_grid(visdir / "gradcam_grid.png", records, visdir)
    imagenet_model = None
    if mode in {"both", "eyeball"}:
        from torchvision.models import ResNet50_Weights, resnet50
        imagenet_weights = ResNet50_Weights.DEFAULT
        imagenet_model = resnet50(weights=imagenet_weights).to(device).eval()
        eye_dir = out / "eyeball"; eye_dir.mkdir(parents=True, exist_ok=True)
        imagenet_rows = []
        for row, reason in selected:
            orig = open_rgb(args.data_root / row["path"])
            label_info = by_id[row["image_id"]]
            generic, generic_conf = imagenet_predict(orig, model=imagenet_model, weights=imagenet_weights)
            imagenet_rows.append({"image_id": row["image_id"], "path": row["path"], "true_label": row["label"],
                                  "imagenet_top1": generic, "imagenet_confidence": generic_conf,
                                  "finetuned_prediction": label_info["predicted_label"],
                                  "finetuned_confidence": float(label_info["confidence"]),
                                  "correct": int(label_info["true_label"] == label_info["predicted_label"]),
                                  "selection_reason": reason})
        save_csv(eye_dir / "error_analysis.csv", imagenet_rows)
        _eyeball_grid(eye_dir / "eyeball_grid.png", imagenet_rows, args.data_root)
    _log_analysis_outputs(args, out, run, mode)
    save_json(visdir / "example_manifest.json", records)
    return {"architecture": arch, "seed": seed, "examples": len(selected), "gradcam": mode in {"both", "gradcam"}, "eyeball": mode in {"both", "eyeball"}}


def _heat_image(array: np.ndarray, size: tuple[int, int]) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps
    values = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    rgba = (colormaps["jet"](np.clip(values, 0, 1))[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGB").resize(size, Image.Resampling.BILINEAR)


def _font(size=13):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _cam_grid(path: Path, rows, directory: Path):
    cell_w, cell_h = 690, 260
    canvas = Image.new("RGB", (cell_w * 3, cell_h * max(1, len(rows))), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        y = i * cell_h
        for j, key in enumerate(("original", "heatmap", "overlay")):
            im = Image.open(directory / row[key]).convert("RGB"); im.thumbnail((cell_w - 12, 215), Image.Resampling.LANCZOS)
            canvas.paste(im, (j * cell_w + 5, y + 2))
            draw.text((j * cell_w + 5, y + 222), f"{key}: {row['image_id']}", fill="black", font=_font(12))
        draw.text((5, y + 242), f"true={row['true_label']} predicted={row['predicted_label']} confidence={row['confidence']:.3f} ({row['selection_reason']})", fill="black", font=_font(12))
    canvas.save(path)


def _eyeball_grid(path: Path, rows, root: Path):
    width, height = 450, 285
    canvas = Image.new("RGB", (width * 3, height * max(1, (len(rows) + 2) // 3)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x, y = (i % 3) * width, (i // 3) * height
        im = open_rgb(root / row["path"]); im.thumbnail((width - 12, 195), Image.Resampling.LANCZOS)
        canvas.paste(im, (x + 5, y + 3))
        draw.text((x + 5, y + 205), f"true={row['true_label']} | ImageNet={row['imagenet_top1']}", fill="black", font=_font(12))
        draw.text((x + 5, y + 223), f"tuned={row['finetuned_prediction']} ({row['finetuned_confidence']:.3f})", fill="black", font=_font(12))
        draw.text((x + 5, y + 242), f"{'correct' if row['correct'] else 'ERROR'} | {row['selection_reason']}", fill="black", font=_font(11))
    canvas.save(path)


def _log_analysis_outputs(args, out: Path, selected_run: dict, mode: str) -> None:
    status = {"requested_mode": args.wandb_mode, "logged": False}
    if args.wandb_mode == "disabled":
        status["reason"] = "W&B disabled by request"
    else:
        try:
            import wandb
            arch, seed = selected_run["architecture"], selected_run["seed"]
            tables = {}
            for name, file in (("architecture_summary", out / "final" / "architecture_summary.csv"),
                               ("statistical_tests", out / "final" / "statistical_tests.csv"),
                               ("per_class_summary", out / "final" / "per_class_summary.csv")):
                if file.exists():
                    with file.open("r", encoding="utf-8-sig", newline="") as stream:
                        records = list(csv.DictReader(stream))
                    if records:
                        columns = list(records[0])
                        tables[name] = wandb.Table(columns=columns, data=[[r.get(k, "") for k in columns] for r in records])
            run = wandb.init(project=args.wandb_project, group="final", name=f"final_visual_analysis_{arch}_seed{seed}",
                             entity=args.wandb_entity, mode=args.wandb_mode, reinit=True,
                             config={"stage": "final_analysis", "architecture": arch, "seed": seed,
                                     "visuals": mode})
            artifacts = {}
            if mode in {"both", "gradcam"} and (out / "gradcam" / "gradcam_grid.png").exists():
                artifacts["gradcam_grid"] = wandb.Image(str(out / "gradcam" / "gradcam_grid.png"))
            if mode in {"both", "eyeball"} and (out / "eyeball" / "eyeball_grid.png").exists():
                artifacts["eyeball_grid"] = wandb.Image(str(out / "eyeball" / "eyeball_grid.png"))
            confusion = Path(selected_run["run_dir"]) / "confusion_matrix.png"
            if confusion.exists():
                artifacts["confusion_matrix"] = wandb.Image(str(confusion))
            artifacts.update(tables)
            if artifacts:
                run.log(artifacts)
            run.finish()
            status.update({"logged": True, "run_url": getattr(run, "url", None)})
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
    save_json(out / "report_assets" / "analysis_wandb_status.json", status)

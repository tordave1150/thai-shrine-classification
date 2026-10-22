from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50

from .data import CLASSES
from .training import ImageDataset, load_split
from .analysis import save_csv, save_json


def imagenet_predict(image: Image.Image, model=None, weights=None) -> tuple[str, float]:
    if weights is None:
        weights = ResNet50_Weights.DEFAULT
    if model is None:
        model = resnet50(weights=weights)
        model.eval()
    tensor = weights.transforms()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = model(tensor).softmax(dim=1)[0]
    index = int(probs.argmax())
    return str(weights.meta["categories"][index]), float(probs[index])


def run_baseline(args, out: Path) -> dict:
    rows = [r for r in load_split(out / "splits.csv") if r["split"] == "train"]
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    dataset = ImageDataset(args.data_root, rows, training=False, size=args.input_size)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=args.num_workers)
    results = []
    with torch.no_grad():
        for tensors, labels, ids, _ in loader:
            logits = model(tensors.to(next(model.parameters()).device)).softmax(dim=1)
            for i, image_id in enumerate(ids):
                pred = int(logits[i].argmax())
                row = next(x for x in rows if x["image_id"] == image_id)
                results.append({"image_id": image_id, "true_custom_class": row["label"],
                                "imagenet_top1": weights.meta["categories"][pred],
                                "imagenet_confidence": float(logits[i, pred])})
    outdir = out / "baseline"
    save_csv(outdir / "pretrained_predictions.csv", results)
    save_json(outdir / "baseline_metadata.json", {"model": "resnet50", "weights": "ResNet50_Weights.DEFAULT",
                                                   "interpretation": "qualitative 1000-class label-space mismatch only; no shrine-class metrics"})
    _grid(args.data_root, rows, {r["image_id"]: r for r in results}, outdir / "pretrained_examples.png")
    try:
        import wandb
        run = wandb.init(project=args.wandb_project, group="baseline", name="baseline_imagenet_resnet50",
                         entity=args.wandb_entity, mode=args.wandb_mode, reinit=True)
        run.log({"baseline_examples": wandb.Image(str(outdir / "pretrained_examples.png"))})
        run.finish()
    except Exception as exc:
        save_json(outdir / "wandb_status.json", {"mode": args.wandb_mode, "error": f"{type(exc).__name__}: {exc}", "local_saved": True})
    return {"images": len(results), "output": str(outdir)}


def _grid(root: Path, rows, results, path: Path):
    from .data import open_rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (4 * 330, 3 * 260), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    selected = []
    for cls in CLASSES:
        selected.extend([r for r in rows if r["label"] == cls][:3])
    for i, row in enumerate(selected[:12]):
        x, y = (i % 4) * 330, (i // 4) * 260
        im = open_rgb(root / row["path"]); im.thumbnail((310, 205), Image.Resampling.LANCZOS)
        canvas.paste(im, (x + 8, y + 4))
        pred = results[row["image_id"]]
        draw.text((x + 6, y + 212), f"true: {row['label']}", fill="black", font=font)
        draw.text((x + 6, y + 231), f"ImageNet: {pred['imagenet_top1']} ({pred['imagenet_confidence']:.2f})", fill="black", font=font)
    canvas.save(path)

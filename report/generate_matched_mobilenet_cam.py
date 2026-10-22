"""Generate only eight MobileNet CAMs matched to the saved ResNet cases.

Explicit user authorization: 22 October 2026. No training or tuning.
Run from the project root: .venv/Scripts/python.exe report/generate_matched_mobilenet_cam.py
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import uuid
from datetime import datetime, timezone
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from PIL import Image
from torchvision import transforms
from pipeline import _check_existing_split
from pipeline_modes import _read_csv, _csv, _json
from resnet_followup import locate
from shrine_classifier.data import CLASSES, open_rgb, letterbox, sha256_file, IMAGENET_MEAN, IMAGENET_STD
from shrine_classifier.models import build_model
from shrine_classifier.analysis import gradcam
from shrine_classifier.training import seed_everything


def main():
    out = ROOT / "outputs_modes_v3_td"
    data = ROOT / "dataset_cleaned"
    _check_existing_split(out, data)
    registry = json.loads((out / "run_registry.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert selection["reference_pair"] == {"architecture": "mobilenet_v3_large", "mode": "partial"}
    matching = [r for r in registry["final_runs"] if r["status"] == "completed"
                and r["architecture"] == "mobilenet_v3_large" and r["mode"] == "partial" and int(r["seed"]) == 42]
    if len(matching) != 1:
        raise ValueError("Expected exactly one completed MobileNet/Partial seed-42 run")
    run = matching[0]
    checkpoint = locate(out, run["checkpoint"])
    run_dir = locate(out, run["run_dir"])
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cases_path = ROOT / "report/resnet_gradcam_cases.csv"
    cases = _read_csv(cases_path)
    if len(cases) != 8 or len({r["image_id"] for r in cases}) != 8:
        raise ValueError("Expected eight fixed ResNet case IDs")
    preds_path = run_dir / "test_predictions.csv"
    preds = {r["image_id"]: r for r in _read_csv(preds_path)}
    split = {r["image_id"]: r for r in _read_csv(out / "splits.csv") if r["split"] == "test"}
    if len(preds) != 85 or set(preds) != set(split):
        raise ValueError("Saved prediction IDs do not match canonical test split")
    parents = out / "gradcam/mobilenet_v3_large_partial_matched_resnet/seed42"
    parents.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint_sha256": sha256_file(checkpoint), "case_selection_sha256": sha256_file(cases_path),
                "prediction_sha256": sha256_file(preds_path), "selection_sha256": sha256_file(out / "selection.json")}
    latest = parents / "latest.json"
    if latest.exists():
        previous = json.loads(latest.read_text(encoding="utf-8"))
        if previous["inputs"] != identity:
            raise ValueError("Existing matched-CAM inputs differ; inspect before repeating")
        for relative, digest in previous["artifact_sha256"].items():
            if sha256_file(out / relative) != digest:
                raise ValueError("Existing matched-CAM artifact changed")
        print("Existing complete eight-case matched CAM retained.")
        return
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model("mobilenet_v3_large", pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (state["architecture"], state["mode"], int(state["seed"]), int(state["epoch"])) != ("mobilenet_v3_large", "partial", 42, int(run["best_epoch"])):
        raise ValueError("Checkpoint identity mismatch")
    model.load_state_dict(state["model_state"], strict=True)
    model.to(device).eval()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dest = parents / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    dest.mkdir(exist_ok=False)
    records = []
    for case in cases:
        image_id = case["image_id"]
        row, pred = split[image_id], preds[image_id]
        if row["label"] != case["true_label"] or pred["true_label"] != row["label"]:
            raise ValueError("Case label mismatch")
        original = open_rgb(data / row["path"])
        canvas, bbox = letterbox(original, int(cfg["input_size"]))
        tensor = transform(canvas).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = model(tensor).softmax(1)[0].cpu().numpy()
        saved = np.asarray([float(pred["prob_" + c]) for c in CLASSES])
        error = float(np.abs(probabilities - saved).max())
        if probabilities.argmax() != saved.argmax():
            _json(dest / "failure.json", {"image_id": image_id, "completed_cases": len(records), "error": "predicted class changed", "max_probability_error": error})
            raise ValueError(f"Prediction changed for {image_id}: max probability error {error}")
        if error > 1e-3:
            print(f"[note] {image_id}: same predicted class; saved batched and current single-image probabilities differ by {error:.6f}. Both are retained in metadata.", flush=True)
        with torch.enable_grad():
            heat = gradcam(model, "mobilenet_v3_large", tensor, CLASSES.index(pred["predicted_label"]))[0].detach().cpu().numpy()
        x1, y1, x2, y2 = bbox
        crop = heat[y1:y2, x1:x2]
        if not np.isfinite(crop).all() or not crop.size or not np.any(crop > 0):
            raise ValueError("Empty/nonfinite CAM: " + image_id)
        aligned = np.asarray(Image.fromarray(crop.astype(np.float32)).resize(original.size, Image.Resampling.BILINEAR))
        color = Image.fromarray(np.uint8(colormaps["jet"](np.clip(aligned, 0, 1))[..., :3] * 255))
        sample = dest / image_id.replace("/", "__")
        sample.mkdir()
        original.save(sample / "original.png")
        color.save(sample / "heatmap.png")
        Image.blend(original, color, .42).save(sample / "overlay.png")
        np.save(sample / "cam_original_alignment.npy", aligned)
        records.append({**pred, "explained_target": pred["predicted_label"], "cam_layer": "features[-1]",
                        "selection_reason": "same image ID and order as the saved ResNet50/Full eight cases",
                        "content_bbox": json.dumps(list(bbox)), "checkpoint_sha256": identity["checkpoint_sha256"],
                        "probability_max_abs_error": error, "cam_inference_confidence": float(probabilities.max()),
                        "cam_inference_probabilities": json.dumps(probabilities.tolist()), "sample_directory": sample.relative_to(out).as_posix()})
        print(f"[{len(records)}/8] {image_id}: {pred['predicted_label']}, saved confidence={float(pred['confidence']):.6f}, error={error:.2e}", flush=True)
    _csv(dest / "selected_cases.csv", records)
    artifact_hashes = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    manifest = {"architecture": "mobilenet_v3_large", "mode": "partial", "seed": 42, "examples": 8,
                "inputs": identity, "directory": dest.relative_to(out).as_posix(), "artifact_sha256": artifact_hashes,
                "device": device, "torch": torch.__version__, "training_runs_added": 0,
                "case_ids": [c["image_id"] for c in cases], "explained_target": "each model's own predicted class",
                "created_utc": datetime.now(timezone.utc).isoformat()}
    _json(dest / "manifest.json", manifest)
    _json(latest, manifest)
    print(json.dumps({"completed_cases": 8, "training_runs_added": 0, "latest": str(latest)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["san_phraphrom", "san_phraphum", "san_tayay", "tee_ju_eia"]
CLASS_TH = ["ศาลพระพรหม", "ศาลพระภูมิ", "ศาลตายาย", "ศาลตี่จู้เอี๊ยะ"]
NEUTRAL = (124, 116, 104)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_rgb(path: Path) -> Image.Image:
    """Open one image/frame, orient it, and composite transparency neutrally."""
    with Image.open(path) as image:
        image.seek(0)
        oriented = ImageOps.exif_transpose(image)
        if oriented.mode == "CMYK":
            converted = oriented.convert("RGB")
        elif "A" in oriented.getbands() or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, NEUTRAL + (255,))
            converted = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            converted = oriented.convert("RGB")
        converted.load()
        return converted.copy()


def letterbox(image: Image.Image, size: int = 224) -> tuple[Image.Image, tuple[int, int, int, int]]:
    image = image.convert("RGB")
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    x = (size - width) // 2
    y = (size - height) // 2
    canvas = Image.new("RGB", (size, size), NEUTRAL)
    canvas.paste(resized, (x, y))
    return canvas, (x, y, x + width, y + height)


def _phash(image: Image.Image) -> int:
    from scipy.fft import dctn

    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = dctn(gray, norm="ortho")[:8, :8].flatten()[1:]
    median = float(np.median(low))
    value = 0
    for coefficient in low:
        value = (value << 1) | int(coefficient > median)
    return value


def _distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _load_metadata(root: Path, *, required: bool) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, Any]]:
    path = root / "metadata_cleaned.csv"
    if not path.exists():
        if not required:
            return {}, {"path": None, "rows": 0, "rows_with_url": 0}
        raise FileNotFoundError(f"Required provenance file not found: {path}")
    by_id: dict[tuple[str, str], dict[str, str]] = {}
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "category", "source_url"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            rows += 1
            key = (row["category"].strip(), Path(row["filename"].strip()).stem)
            if key in by_id:
                raise ValueError(f"Duplicate metadata image ID: {key}")
            by_id[key] = {k: (v or "").strip() for k, v in row.items() if k}
    return by_id, {"path": str(path), "sha256": sha256_file(path), "rows": rows,
                   "rows_with_url": sum(bool(r.get("source_url")) for r in by_id.values())}


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def audit_dataset(root: Path, out: Path, *, near_threshold: int = 6,
                  require_metadata: bool = True) -> dict[str, Any]:
    root, out = root.resolve(), out.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    out.mkdir(parents=True, exist_ok=True)
    metadata, metadata_summary = _load_metadata(root, required=require_metadata)
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.relative_to(root).as_posix().casefold())
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    unexpected = [p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() not in EXTENSIONS and p.name != "metadata_cleaned.csv"]
    for path in paths:
        rel = path.relative_to(root)
        label = rel.parts[0] if rel.parts else ""
        if label not in CLASSES:
            errors.append({"path": rel.as_posix(), "error": f"unknown class directory: {label}"})
            continue
        key = (label, path.stem)
        try:
            image = open_rgb(path)
            with Image.open(path) as raw_image:
                original_mode = raw_image.mode
                original_format = raw_image.format or "unknown"
                transparency = "A" in raw_image.getbands() or "transparency" in raw_image.info
                frame_count = int(getattr(raw_image, "n_frames", 1))
            raw = sha256_file(path)
            pixels = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
            meta = metadata.get(key, {})
            if meta and meta.get("category") != label:
                raise ValueError(f"metadata class {meta.get('category')} conflicts with path class {label}")
            url = meta.get("source_url", "")
            domain = urlsplit(url).hostname or "" if url else ""
            pixels_array = np.asarray(image, dtype=np.uint8)
            nearblack = np.all(pixels_array <= 10, axis=2)
            band_x, band_y = max(1, image.width // 50), max(1, image.height // 50)
            edge = np.zeros(nearblack.shape, dtype=bool)
            edge[:band_y] = edge[-band_y:] = True
            edge[:, :band_x] = edge[:, -band_x:] = True
            rows.append({
                "image_id": f"{label}/{path.stem}", "path": rel.as_posix(), "label": label,
                "label_index": CLASSES.index(label), "width": image.width, "height": image.height,
                "mode": original_mode, "format": original_format,
                "has_alpha": int(transparency), "animated_frames": frame_count,
                "byte_sha256": raw, "pixel_sha256": pixels, "source_url": url, "source_domain": domain,
                "phash": f"{_phash(image):016x}", "file_bytes": path.stat().st_size,
                "nearblack_fraction": float(nearblack.mean()),
                "edge_nearblack_fraction": float(nearblack[edge].mean()),
            })
        except Exception as exc:
            errors.append({"path": rel.as_posix(), "error": f"{type(exc).__name__}: {exc}"})

    # Detect metadata/image mismatches without treating .jpg/.png as different source IDs.
    actual_ids = {(r["label"], Path(r["path"]).stem) for r in rows}
    metadata_only = sorted(f"{k[0]}/{k[1]}" for k in set(metadata) - actual_ids)
    image_only = sorted(f"{k[0]}/{k[1]}" for k in actual_ids - set(metadata))
    id_collisions: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        id_collisions[row["image_id"]].append(row["path"])
    collisions = {k: v for k, v in id_collisions.items() if len(v) > 1}

    exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    url_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        exact[row["pixel_sha256"]].append(row)
        if row["source_url"]:
            url_groups[row["source_url"]].append(row["image_id"])
    exact_groups = [v for v in exact.values() if len(v) > 1]
    repeated_urls = [v for v in url_groups.values() if len(v) > 1]
    if require_metadata and (metadata_only or image_only or collisions):
        errors.append({"path": "metadata_cleaned.csv", "error":
                       f"metadata join mismatch: metadata_only={len(metadata_only)}, image_only={len(image_only)}, ID collisions={len(collisions)}"})
    missing_classes = {c: sum(r["label"] == c for r in rows) for c in CLASSES}
    for c, count in missing_classes.items():
        if count == 0:
            errors.append({"path": str(root), "error": f"missing/empty class: {c}"})

    hashes = [(r, int(r["phash"], 16)) for r in rows]
    candidates: list[dict[str, Any]] = []
    for i, (a, ha) in enumerate(hashes):
        for b, hb in hashes[i + 1:]:
            distance = _distance(ha, hb)
            if distance <= near_threshold:
                candidates.append({"path_a": a["path"], "path_b": b["path"],
                                   "id_a": a["image_id"], "id_b": b["image_id"],
                                   "cross_class": int(a["label"] != b["label"]),
                                   "hamming_distance": distance, "review_status": "pending"})
    candidates.sort(key=lambda r: (r["hamming_distance"], r["path_a"], r["path_b"]))

    review_path = out / "near_duplicate_review_decisions.csv"
    review_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    stale_review_rows = []
    if review_path.exists():
        with review_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for reviewed in csv.DictReader(stream):
                pair = tuple(sorted((reviewed.get("id_a", ""), reviewed.get("id_b", ""))))
                if pair not in {tuple(sorted((r["id_a"], r["id_b"]))) for r in candidates}:
                    stale_review_rows.append(reviewed)
                    continue
                review_by_pair[pair] = reviewed
        for candidate in candidates:
            prior = review_by_pair.get(tuple(sorted((candidate["id_a"], candidate["id_b"]))))
            if prior:
                candidate["review_status"] = prior.get("review_status", "pending")

    _write_csv(out / "image_properties.csv", rows)
    _write_csv(out / "near_duplicate_candidates.csv", candidates)
    _write_csv(out / "duplicate_groups.csv", [
        {"group_type": "exact_pixels", "image_ids": ";".join(r["image_id"] for r in group),
         "paths": ";".join(r["path"] for r in group), "review_status": "confirmed_exact"}
        for group in exact_groups
    ] + [
        {"group_type": "source_url", "image_ids": ";".join(ids), "paths": "", "review_status": "confirmed_url"}
        for ids in repeated_urls
    ] + [
        {"group_type": "visual_near_duplicate", "image_ids": ";".join((r["id_a"], r["id_b"])),
         "paths": ";".join((r["path_a"], r["path_b"])), "review_status": "confirmed_same_source"}
        for r in candidates if r["review_status"] == "confirmed_same_source"
    ])
    _write_csv(out / "metadata_join_issues.csv", [
        *({"issue": "metadata_only", "image_id": x} for x in metadata_only),
        *({"issue": "image_only", "image_id": x} for x in image_only),
    ])
    _write_csv(out / "audit_errors.csv", errors)
    summary = {
        "dataset_root": str(root), "audit_time_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "class_order": CLASSES, "num_images": len(rows), "num_read_errors_or_identity_issues": len(errors),
        "class_counts": missing_classes, "mode_counts": dict(Counter(r["mode"] for r in rows)),
        "format_counts": dict(Counter(r["format"] for r in rows)),
        "distinct_dimensions": len({(r["width"], r["height"]) for r in rows}),
        "dimension_range": {"width": [min((r["width"] for r in rows), default=0), max((r["width"] for r in rows), default=0)],
                            "height": [min((r["height"] for r in rows), default=0), max((r["height"] for r in rows), default=0)]},
        "alpha_or_transparency_count": sum(r["has_alpha"] for r in rows),
        "all_or_nearly_all_black_files": [r["path"] for r in rows if r["nearblack_fraction"] >= 0.98],
        "mean_nearblack_fraction_by_class": {c: float(np.mean([r["nearblack_fraction"] for r in rows if r["label"] == c])) for c in CLASSES},
        "mean_edge_nearblack_fraction_by_class": {c: float(np.mean([r["edge_nearblack_fraction"] for r in rows if r["label"] == c])) for c in CLASSES},
        "animated_file_count": sum(r["animated_frames"] > 1 for r in rows),
        "exact_decoded_pixel_duplicate_groups": [[r["image_id"] for r in g] for g in exact_groups],
        "repeated_source_url_groups": repeated_urls, "num_near_duplicate_candidates": len(candidates),
        "near_duplicate_review": {"file": review_path.name if review_path.exists() else None,
                                  "confirmed": sum(r["review_status"] == "confirmed_same_source" for r in candidates),
                                  "rejected": sum(r["review_status"] == "rejected_different_image" for r in candidates),
                                  "pending": sum(r["review_status"] == "pending" for r in candidates),
                                  "stale_rows": len(stale_review_rows)},
        "phash_hamming_threshold": near_threshold,
        "metadata": {**metadata_summary, "status": "checked" if require_metadata else "not_supplied_for_this_version"},
        "source_domain_counts_by_class": {c: dict(Counter(r["source_domain"] or "unknown" for r in rows if r["label"] == c)) for c in CLASSES},
        "metadata_only_ids": metadata_only, "image_only_ids": image_only if require_metadata else [],
        "unexpected_files": unexpected, "errors": errors,
        "limitations": ["Perceptual hash rows are candidates, not confirmed duplicates.",
                        "Visual/source-site shortcut review remains a human review task."],
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_distribution(out / "class_distribution.png", missing_classes)
    _save_sample_grid(root, rows, out / "sample_grid.png")
    _save_candidate_grid(root, candidates[:100], out / "near_duplicate_review.png")
    _save_preprocessing_grid(root, rows, out / "preprocessing_examples.png")
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["issue", "image_id"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _font(size: int = 18):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _save_distribution(path: Path, counts: dict[str, int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(CLASSES, [counts[c] for c in CLASSES], color=["#4178be", "#47a881", "#f0ad4e", "#d95d6a"])
    ax.set_ylabel("Verified image count")
    ax.set_title("Thai Shrine Dataset - cleaned source")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_sample_grid(root: Path, rows: list[dict[str, Any]], path: Path) -> None:
    cell_w, thumb = 260, (242, 170)
    canvas = Image.new("RGB", (4 * cell_w, 690), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(CLASSES):
        draw.text((col * cell_w + 8, 4), label, fill="black", font=_font(16))
        chosen = [r for r in rows if r["label"] == label][:3]
        for ridx, row in enumerate(chosen):
            image = open_rgb(root / row["path"])
            image.thumbnail(thumb, Image.Resampling.LANCZOS)
            x = col * cell_w + (cell_w - image.width) // 2
            y = 30 + ridx * 215
            canvas.paste(image, (x, y))
            draw.text((col * cell_w + 5, y + 174), row["image_id"], fill="black", font=_font(11))
    canvas.save(path)


def _save_preprocessing_grid(root: Path, rows: list[dict[str, Any]], path: Path) -> None:
    canvas = Image.new("RGB", (4 * 440, 285), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, label in enumerate(CLASSES):
        row = next((r for r in rows if r["label"] == label), None)
        if not row:
            continue
        original = open_rgb(root / row["path"])
        original.thumbnail((205, 220), Image.Resampling.LANCZOS)
        fitted, bbox = letterbox(open_rgb(root / row["path"]), 224)
        x = idx * 440
        canvas.paste(original, (x + 4, 30))
        canvas.paste(fitted, (x + 218, 30))
        draw.text((x + 4, 5), f"{label} - original", fill="black", font=_font(14))
        draw.text((x + 218, 5), f"neutral letterbox, bbox={bbox}", fill="black", font=_font(12))
        draw.text((x + 4, 255), row["path"], fill="black", font=_font(12))
    canvas.save(path)


def _save_candidate_grid(root: Path, candidates: list[dict[str, Any]], path: Path) -> None:
    cell_w, cell_h, thumb = 420, 150, (180, 115)
    n = max(1, len(candidates))
    canvas = Image.new("RGB", (2 * cell_w, ((n + 1) // 2) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, item in enumerate(candidates):
        x, y = (idx % 2) * cell_w, (idx // 2) * cell_h
        for offset, key in ((0, "path_a"), (200, "path_b")):
            img = open_rgb(root / item[key])
            img.thumbnail(thumb, Image.Resampling.LANCZOS)
            canvas.paste(img, (x + offset, y))
        draw.text((x, y + 118), f"d={item['hamming_distance']} {item['id_a']} <> {item['id_b']}", fill="black", font=_font(11))
    canvas.save(path)


def groups_from_audit(rows: list[dict[str, str]], reviewed: Path | None = None) -> dict[str, str]:
    uf = _UnionFind()
    by_pixels: dict[str, list[str]] = defaultdict(list)
    by_url: dict[str, list[str]] = defaultdict(list)
    labels = {r["image_id"]: r["label"] for r in rows}
    for r in rows:
        uf.find(r["image_id"])
        by_pixels[r["pixel_sha256"]].append(r["image_id"])
        if r.get("source_url"):
            by_url[r["source_url"]].append(r["image_id"])
    for collection in (by_pixels.values(), by_url.values()):
        for ids in collection:
            for other in ids[1:]:
                uf.union(ids[0], other)
    if reviewed and reviewed.exists():
        with reviewed.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("review_status") == "confirmed_same_source":
                    a, b = row["id_a"], row["id_b"]
                    if a not in labels or b not in labels:
                        raise ValueError(f"Duplicate review references unknown IDs: {a}, {b}")
                    uf.union(a, b)
    members: dict[str, list[str]] = defaultdict(list)
    for image_id in labels:
        members[uf.find(image_id)].append(image_id)
    conflict = [ids for ids in members.values() if len({labels[i] for i in ids}) > 1]
    if conflict:
        raise ValueError(f"Confirmed duplicate group has conflicting labels: {conflict[0]}")
    return {image_id: min(ids) for ids in members.values() for image_id in ids}


def create_split(data_root: Path, audit_dir: Path, out_csv: Path, *, split_seed: int = 42,
                 review_csv: Path | None = None) -> dict[str, Any]:
    if out_csv.exists():
        raise FileExistsError(f"Refusing to overwrite canonical split {out_csv}; inspect it or choose a new --out-dir")
    props = _read_csv(audit_dir / "image_properties.csv")
    if not props:
        raise ValueError("Run --stage audit before --stage prep")
    groups = groups_from_audit(props, review_csv)
    ids = np.array([r["image_id"] for r in props])
    y = np.array([r["label"] for r in props])
    g = np.array([groups[i] for i in ids])
    classes = np.array(CLASSES)
    # Deterministic repeated grouped holdouts; choose the candidate closest to all targets.
    outer = GroupShuffleSplit(n_splits=256, test_size=0.15, random_state=split_seed)
    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for temp_idx, test_idx in outer.split(ids, y, groups=g):
        inner = GroupShuffleSplit(n_splits=12, test_size=0.15 / 0.85, random_state=split_seed)
        for train_rel, val_rel in inner.split(ids[temp_idx], y[temp_idx], groups=g[temp_idx]):
            train_idx, val_idx = temp_idx[train_rel], temp_idx[val_rel]
            splits = (train_idx, val_idx, test_idx)
            if any(set(y[ix]) != set(CLASSES) for ix in splits):
                continue
            if any(set(g[splits[i]]) & set(g[splits[j]]) for i, j in ((0, 1), (0, 2), (1, 2))):
                continue
            score = 0.0
            for ix, target in zip(splits, (0.70, 0.15, 0.15)):
                score += ((len(ix) / len(ids) - target) / target) ** 2
                for cls in CLASSES:
                    observed = np.sum(y[ix] == cls) / np.sum(y == cls)
                    score += 0.35 * ((observed - target) / target) ** 2
            if best is None or score < best[0]:
                best = (score, train_idx, val_idx, test_idx)
    if best is None:
        raise ValueError("Could not build a group-stratified split with all four classes; review duplicate groups or dataset labels")
    _, train_idx, val_idx, test_idx = best
    assignment = {ids[i]: split for split, ix in (("train", train_idx), ("validation", val_idx), ("test", test_idx)) for i in ix}
    lookup = {r["image_id"]: r for r in props}
    records = [{"image_id": image_id, "path": lookup[image_id]["path"], "label": lookup[image_id]["label"],
                "label_index": int(lookup[image_id]["label_index"]), "group_id": groups[image_id], "split": assignment[image_id]}
               for image_id in ids]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_csv, records)
    manifest_digest = hashlib.sha256("\n".join(f"{r['image_id']}:{r['pixel_sha256']}" for r in props).encode()).hexdigest()
    group_digest = hashlib.sha256("\n".join(f"{k}:{groups[k]}" for k in sorted(groups)).encode()).hexdigest()
    split_digest = hashlib.sha256("\n".join(f"{r['image_id']}:{r['split']}" for r in sorted(records, key=lambda x: x["image_id"])).encode()).hexdigest()
    summary = {"split_seed": split_seed, "method": "256x12 deterministic GroupShuffleSplit candidates; minimize split-size and per-class deviations",
               "target_ratios": {"train": .70, "validation": .15, "test": .15},
               "counts": {name: {"n": sum(r["split"] == name for r in records),
                                 "by_class": {c: sum(r["split"] == name and r["label"] == c for r in records) for c in CLASSES}}
                          for name in ("train", "validation", "test")},
               "group_overlap": False, "num_groups": len(set(groups.values())), "class_order": CLASSES,
               "manifest_sha256": manifest_digest, "group_sha256": group_digest, "split_sha256": split_digest,
               "duplicate_review_sha256": sha256_file(review_csv) if review_csv and review_csv.exists() else None,
               "human_near_duplicate_review": str(review_csv) if review_csv else "not supplied; perceptual candidates remain unconfirmed"}
    meta_path = out_csv.parent / "split_metadata.json"
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

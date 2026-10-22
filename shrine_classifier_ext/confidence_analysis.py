"""Item 1 — Confidence / reliability analysis.

Reads canonical final test_predictions.csv files (one per architecture × seed),
validates probability integrity, computes ECE and bin metrics, and generates
reliability diagrams and confidence histograms.

No model inference is performed here; only cached CSV predictions are loaded.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from shrine_classifier.data import CLASSES
from shrine_classifier.analysis import save_csv, save_json

N_CLASSES = len(CLASSES)
BIN_EDGES = np.linspace(0.0, 1.0, 11)  # 10 equal-width bins [0,.1),...,[.9,1]


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def _load_predictions_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _validate_predictions(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Validate a predictions CSV and return enriched records.

    Raises ValueError describing the first integrity failure found.
    """
    issues: list[str] = []
    seen_ids: set[str] = set()
    clean: list[dict[str, Any]] = []
    prob_cols = [f"prob_{c}" for c in CLASSES]

    for idx, row in enumerate(rows):
        image_id = row.get("image_id", "")
        if not image_id:
            issues.append(f"row {idx}: missing image_id")
            continue
        if image_id in seen_ids:
            issues.append(f"row {idx}: duplicate image_id '{image_id}'")
            continue
        seen_ids.add(image_id)

        true_label = row.get("true_label", "")
        if true_label not in CLASSES:
            issues.append(f"row {idx}: unknown true_label '{true_label}'")
            continue

        predicted_label = row.get("predicted_label", "")
        if predicted_label not in CLASSES:
            issues.append(f"row {idx}: unknown predicted_label '{predicted_label}'")
            continue

        try:
            confidence = float(row.get("confidence", "nan"))
        except (ValueError, TypeError):
            issues.append(f"row {idx}: non-numeric confidence")
            continue

        probs: list[float] = []
        for col in prob_cols:
            try:
                v = float(row.get(col, "nan"))
            except (ValueError, TypeError):
                issues.append(f"row {idx}: non-numeric {col}")
                v = float("nan")
            probs.append(v)

        if any(np.isnan(p) or not np.isfinite(p) or p < 0 or p > 1 for p in probs):
            issues.append(f"row {idx}: probabilities not all finite in [0,1]")
            continue

        prob_sum = sum(probs)
        if abs(prob_sum - 1.0) > 1e-6:
            issues.append(f"row {idx}: probabilities sum to {prob_sum:.8f} (tolerance 1e-6)")
            continue

        # Check argmax consistency with saved predicted_label and confidence
        argmax_idx = int(np.argmax(probs))
        argmax_label = CLASSES[argmax_idx]
        if argmax_label != predicted_label:
            issues.append(
                f"row {idx}: saved predicted_label='{predicted_label}' does not match "
                f"argmax='{argmax_label}'"
            )
            continue
        max_prob = probs[argmax_idx]
        if abs(max_prob - confidence) > 1e-6:
            issues.append(
                f"row {idx}: saved confidence={confidence:.8f} does not match max_prob={max_prob:.8f}"
            )
            continue

        true_idx = CLASSES.index(true_label)
        correct = int(argmax_idx == true_idx)
        clean.append({
            "image_id": image_id,
            "true_label": true_label,
            "true_index": true_idx,
            "predicted_label": predicted_label,
            "predicted_index": argmax_idx,
            "confidence": confidence,
            "correct": correct,
            "probs": probs,
        })

    if issues:
        summary = "\n".join(issues[:20])
        extra = len(issues) - 20
        if extra > 0:
            summary += f"\n... and {extra} more issues"
        raise ValueError(f"Prediction integrity failure in {source}:\n{summary}")

    return clean


# ---------------------------------------------------------------------------
# Bin metrics and ECE
# ---------------------------------------------------------------------------

def _bin_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute reliability diagram bin stats over a list of validated records."""
    confidences = np.array([r["confidence"] for r in records])
    corrects = np.array([r["correct"] for r in records])
    n_total = len(records)

    rows: list[dict[str, Any]] = []
    for i in range(len(BIN_EDGES) - 1):
        lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
        if i == len(BIN_EDGES) - 2:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            rows.append({
                "bin_lower": round(lo, 2), "bin_upper": round(hi, 2),
                "count": 0, "mean_confidence": None, "accuracy": None,
                "abs_gap": None,
            })
        else:
            mean_conf = float(confidences[mask].mean())
            acc = float(corrects[mask].mean())
            rows.append({
                "bin_lower": round(lo, 2), "bin_upper": round(hi, 2),
                "count": n_bin, "mean_confidence": mean_conf, "accuracy": acc,
                "abs_gap": abs(acc - mean_conf),
            })
    return rows


def _ece(bin_rows: list[dict[str, Any]], n_total: int) -> float | None:
    if n_total == 0:
        return None
    ece = 0.0
    for b in bin_rows:
        if b["count"] > 0 and b["abs_gap"] is not None:
            ece += (b["count"] / n_total) * b["abs_gap"]
    return ece


# ---------------------------------------------------------------------------
# High-confidence errors
# ---------------------------------------------------------------------------

def _high_confidence_errors(records: list[dict[str, Any]],
                             threshold: float = 0.9) -> list[dict[str, Any]]:
    return [
        {
            "image_id": r["image_id"],
            "true_label": r["true_label"],
            "predicted_label": r["predicted_label"],
            "confidence": r["confidence"],
        }
        for r in records
        if r["correct"] == 0 and r["confidence"] >= threshold
    ]


# ---------------------------------------------------------------------------
# Per-group summaries
# ---------------------------------------------------------------------------

def _group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    correct_confs = [r["confidence"] for r in records if r["correct"] == 1]
    wrong_confs = [r["confidence"] for r in records if r["correct"] == 0]
    return {
        "n_total": len(records),
        "n_correct": len(correct_confs),
        "n_wrong": len(wrong_confs),
        "mean_confidence_correct": float(np.mean(correct_confs)) if correct_confs else None,
        "mean_confidence_wrong": float(np.mean(wrong_confs)) if wrong_confs else None,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_reliability(bin_rows: list[dict[str, Any]], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [b["mean_confidence"] for b in bin_rows if b["count"] > 0 and b["mean_confidence"] is not None]
    ys = [b["accuracy"] for b in bin_rows if b["count"] > 0 and b["accuracy"] is not None]
    counts = [b["count"] for b in bin_rows if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    scatter = ax.scatter(xs, ys, c=counts, cmap="YlOrRd", s=80, zorder=3, label="Bins (sized by count)")
    ax.set_xlabel("Mean max probability (confidence)")
    ax.set_ylabel("Accuracy of predicted class")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    plt.colorbar(scatter, ax=ax, label="Bin count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confidence_hist(records_correct: list[float], records_wrong: list[float],
                           title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(records_correct, bins=BIN_EDGES, alpha=0.65, label=f"Correct (n={len(records_correct)})",
            color="#2196F3", edgecolor="white")
    ax.hist(records_wrong, bins=BIN_EDGES, alpha=0.65, label=f"Wrong (n={len(records_wrong)})",
            color="#F44336", edgecolor="white")
    ax.set_xlabel("Confidence (max softmax probability)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_reliability_pooled(pooled_bin_rows: list[dict[str, Any]],
                              arch: str, n_seeds: int, path: Path) -> None:
    """Plot reliability diagram for per-seed pooled predictions.

    Label clearly that this pools repeated predictions of the same 85 images,
    not independent samples.
    """
    title = (f"Reliability — {arch}\n"
             f"Pooled over {n_seeds} seeds × 85 images (not independent samples)")
    _plot_reliability(pooled_bin_rows, title, path)


# ---------------------------------------------------------------------------
# Discover canonical final prediction CSVs
# ---------------------------------------------------------------------------

def _discover_runs(final_dir: Path) -> list[tuple[str, int, Path]]:
    """Return (architecture, seed, prediction_csv_path) for completed runs."""
    runs: list[tuple[str, int, Path]] = []
    for config_path in sorted(final_dir.glob("*/config.json")):
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("stage") != "final":
            continue
        arch = cfg.get("architecture", "")
        seed = cfg.get("seed", None)
        if not arch or seed is None:
            continue
        pred_path = config_path.parent / "test_predictions.csv"
        if pred_path.is_file():
            runs.append((arch, int(seed), pred_path))
    return runs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_confidence_analysis(base_out_dir: Path, out_dir: Path,
                             selection: dict[str, Any]) -> dict[str, Any]:
    """Analyse confidence/reliability from canonical final test predictions.

    Parameters
    ----------
    base_out_dir: path to primary outputs/ directory containing finals/
    out_dir: outputs_extensions_v1/ directory where extension outputs go
    selection: parsed selection.json dict
    """
    import matplotlib
    matplotlib.use("Agg")

    conf_dir = out_dir / "confidence"
    conf_dir.mkdir(parents=True, exist_ok=True)

    final_dir = base_out_dir / "final"
    runs = _discover_runs(final_dir)
    if not runs:
        raise FileNotFoundError(f"No completed final runs with test_predictions.csv found in {final_dir}")

    architectures = selection.get("architectures", [])
    seeds = selection.get("seeds", [])
    representative_seed = int(selection.get("representative_seed", 42))
    reference_arch = selection.get("reference_architecture", architectures[0] if architectures else "")

    # Per-run records
    run_rows: list[dict[str, Any]] = []
    # Collected for per-architecture aggregation
    arch_ece_map: dict[str, list[float]] = {a: [] for a in architectures}
    all_hce: list[dict[str, Any]] = []
    all_bin_rows: list[dict[str, Any]] = []

    issues_log: list[str] = []

    for arch, seed, pred_path in sorted(runs, key=lambda x: (x[0], x[1])):
        source_label = f"{arch}/seed{seed}"
        try:
            raw = _load_predictions_csv(pred_path)
            records = _validate_predictions(raw, source_label)
        except (OSError, ValueError) as e:
            issues_log.append(str(e))
            run_rows.append({
                "architecture": arch, "seed": seed,
                "status": "invalid", "error": str(e),
            })
            continue

        bin_rows = _bin_metrics(records)
        ece = _ece(bin_rows, len(records))
        stats = _group_stats(records)
        hce = _high_confidence_errors(records, threshold=0.9)
        for h in hce:
            h.update({"architecture": arch, "seed": seed})
        all_hce.extend(hce)

        # Store bin rows for CSV
        for b in bin_rows:
            b_row = {"architecture": arch, "seed": seed, **b}
            all_bin_rows.append(b_row)

        # Per-seed reliability figure (representative seed → main figure)
        fig_path = conf_dir / f"reliability_{arch}_seed{seed}.png"
        seed_title = f"Reliability — {arch} seed {seed}\n(n={len(records)} test images)"
        _plot_reliability(bin_rows, seed_title, fig_path)

        # Per-seed confidence histogram
        correct_confs = [r["confidence"] for r in records if r["correct"] == 1]
        wrong_confs = [r["confidence"] for r in records if r["correct"] == 0]
        hist_path = conf_dir / f"confidence_hist_{arch}_seed{seed}.png"
        _plot_confidence_hist(correct_confs, wrong_confs,
                               f"Confidence distribution — {arch} seed {seed}", hist_path)

        if ece is not None:
            arch_ece_map.setdefault(arch, []).append(ece)

        run_rows.append({
            "architecture": arch, "seed": seed, "status": "ok",
            "n_total": stats["n_total"], "n_correct": stats["n_correct"], "n_wrong": stats["n_wrong"],
            "ece": ece,
            "mean_confidence_correct": stats["mean_confidence_correct"],
            "mean_confidence_wrong": stats["mean_confidence_wrong"],
            "n_high_confidence_errors": len(hce),
        })

    # Architecture summary: per-arch ECE mean ± SD
    arch_summary_rows: list[dict[str, Any]] = []
    for arch in architectures:
        ece_values = arch_ece_map.get(arch, [])
        completed = [r for r in run_rows if r["architecture"] == arch and r.get("status") == "ok"]
        n_correct = sum(r.get("n_correct", 0) for r in completed)
        n_wrong = sum(r.get("n_wrong", 0) for r in completed)
        arch_summary_rows.append({
            "architecture": arch,
            "completed_seeds": len(completed),
            "n_correct_total": n_correct,
            "n_wrong_total": n_wrong,
            "ece_mean": float(np.mean(ece_values)) if ece_values else None,
            "ece_sd": float(np.std(ece_values, ddof=1)) if len(ece_values) >= 2 else None,
            "ece_note": "bin-sensitive on small test set (n=85)",
        })

    # Pooled reliability diagram per architecture (labeled correctly)
    for arch in architectures:
        arch_records_all: list[dict[str, Any]] = []
        arch_seeds = [r for r in run_rows if r["architecture"] == arch and r.get("status") == "ok"]
        for arch_seed_row in arch_seeds:
            # Re-load to get individual records  (needed for pooled plot only)
            seed = arch_seed_row["seed"]
            pred_path_list = [(a2, s2, p) for a2, s2, p in runs if a2 == arch and s2 == seed]
            if not pred_path_list:
                continue
            _, _, pred_path = pred_path_list[0]
            try:
                raw = _load_predictions_csv(pred_path)
                recs = _validate_predictions(raw, f"{arch}/seed{seed}")
                arch_records_all.extend(recs)
            except (OSError, ValueError):
                pass

        if arch_records_all:
            pooled_bins = _bin_metrics(arch_records_all)
            n_seeds = len(arch_seeds)
            pool_fig = conf_dir / f"reliability_{arch}_pooled.png"
            _plot_reliability_pooled(pooled_bins, arch, n_seeds, pool_fig)

            # Representative seed main figure (labeled as single seed)
            rep_seed_rows = [r for r in run_rows if r["architecture"] == arch and r["seed"] == representative_seed and r.get("status") == "ok"]
            if rep_seed_rows:
                rep_pred = [(a2, s2, p) for a2, s2, p in runs if a2 == arch and s2 == representative_seed]
                if rep_pred:
                    _, _, rp = rep_pred[0]
                    try:
                        rraw = _load_predictions_csv(rp)
                        rrecs = _validate_predictions(rraw, f"{arch}/seed{representative_seed}")
                        rep_bins = _bin_metrics(rrecs)
                        main_fig = conf_dir / f"reliability_{arch}.png"
                        _plot_reliability(rep_bins,
                                          f"Reliability — {arch} (representative seed {representative_seed})",
                                          main_fig)
                        rc = [r["confidence"] for r in rrecs if r["correct"] == 1]
                        rw = [r["confidence"] for r in rrecs if r["correct"] == 0]
                        main_hist = conf_dir / f"confidence_hist_{arch}.png"
                        _plot_confidence_hist(rc, rw,
                                              f"Confidence distribution — {arch} seed {representative_seed}",
                                              main_hist)
                    except (OSError, ValueError):
                        pass

    # Save outputs
    save_csv(conf_dir / "run_metrics.csv", run_rows)
    save_csv(conf_dir / "architecture_summary.csv", arch_summary_rows)
    save_csv(conf_dir / "bin_metrics.csv", all_bin_rows)
    save_csv(conf_dir / "high_confidence_errors.csv", all_hce if all_hce else [
        {"note": "No high-confidence errors found (threshold=0.9)"}])

    config_out = {
        "stage": "confidence_analysis",
        "description": "Top-label reliability, ECE and confidence histograms from canonical final predictions",
        "bin_edges": BIN_EDGES.tolist(),
        "high_confidence_threshold": 0.9,
        "multiclass_correctness": "argmax matches true label",
        "ece_formula": "sum_b (n_b/N) * abs(accuracy_b - mean_confidence_b)",
        "pooled_note": ("Pooled plots pool repeated predictions of the same 85 test images "
                        "across seeds; they are NOT from independent images."),
        "architectures": architectures,
        "seeds": seeds,
        "representative_seed": representative_seed,
        "reference_architecture": reference_arch,
        "issues": issues_log,
    }
    save_json(conf_dir / "analysis_config.json", config_out)

    return {
        "run_rows": run_rows,
        "arch_summary": arch_summary_rows,
        "issues": issues_log,
        "out_dir": str(conf_dir),
    }

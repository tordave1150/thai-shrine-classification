"""Item 2 — Training dynamics and tuning analysis.

Reads canonical history.csv files and tuning trial CSVs to produce:
  - Per-run training dynamics table (best/stop epoch, gaps, etc.)
  - Architecture-level summaries
  - Trial inventory reconciled from all_trials.csv (canonical) vs sweep_summary.csv
  - Checkpoint validation against saved metadata
  - Figures: best-vs-stop, gap, and LR/WD scatter vs validation Macro F1

No model loading. Does not compute test metrics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from shrine_classifier.data import CLASSES
from shrine_classifier.analysis import save_csv, save_json


# ---------------------------------------------------------------------------
# History loading
# ---------------------------------------------------------------------------

def _load_history(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load history.csv; separate epoch rows from summary/test-metric rows.

    Returns (epoch_rows, summary_rows).
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.DictReader(f))

    epoch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for row in all_rows:
        raw_epoch = row.get("epoch", "")
        raw_train_loss = row.get("train_loss", "")
        if raw_train_loss and raw_train_loss.strip():
            # This row has a train_loss value → it's an epoch measurement
            try:
                ep = int(float(row["epoch"]))
                train_loss = float(row["train_loss"])
            except (ValueError, TypeError, KeyError):
                summary_rows.append(row)
                continue
            epoch_rows.append({
                "epoch": ep,
                "train_loss": train_loss,
                "train_acc": _safe_float(row.get("train_acc")),
                "train_macro_f1": _safe_float(row.get("train_macro_f1")),
                "val_loss": _safe_float(row.get("val_loss")),
                "val_acc": _safe_float(row.get("val_acc")),
                "val_macro_f1": _safe_float(row.get("val_macro_f1")),
                "learning_rate": _safe_float(row.get("learning_rate")),
            })
        else:
            summary_rows.append(row)

    return epoch_rows, summary_rows


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _validate_epoch_rows(epoch_rows: list[dict[str, Any]], source: str) -> list[str]:
    """Return a list of issue descriptions."""
    issues = []
    epochs_seen = [r["epoch"] for r in epoch_rows]
    if sorted(set(epochs_seen)) != list(range(1, len(epochs_seen) + 1)):
        issues.append(f"{source}: epochs not unique/contiguous: {epochs_seen}")
    required_metrics = ["train_loss", "train_acc", "train_macro_f1",
                        "val_loss", "val_acc", "val_macro_f1"]
    for r in epoch_rows:
        missing = [m for m in required_metrics if r.get(m) is None]
        if missing:
            issues.append(f"{source} epoch {r['epoch']}: missing {missing}")
    return issues


def _best_epoch(epoch_rows: list[dict[str, Any]]) -> int:
    """Return best epoch index by val_macro_f1 → val_loss → earliest epoch."""
    def key(r):
        vf1 = r.get("val_macro_f1") or -999
        vl = r.get("val_loss") or 999
        ep = r.get("epoch", 999)
        return (-vf1, vl, ep)
    return min(epoch_rows, key=key)["epoch"]


# ---------------------------------------------------------------------------
# Dynamics extraction
# ---------------------------------------------------------------------------

def _extract_dynamics(arch: str, seed: int, run_dir: Path) -> dict[str, Any]:
    history_path = run_dir / "history.csv"
    config_path = run_dir / "config.json"
    checkpoint_meta_path = run_dir / "checkpoint_metadata.json"

    result: dict[str, Any] = {"architecture": arch, "seed": seed, "run_dir": str(run_dir)}

    if not history_path.exists():
        result["status"] = "missing_history"
        return result

    epoch_rows, summary_rows = _load_history(history_path)
    if not epoch_rows:
        result["status"] = "empty_history"
        return result

    issues = _validate_epoch_rows(epoch_rows, f"{arch}/seed{seed}")

    # Infer best epoch from history
    best_ep = _best_epoch(epoch_rows)
    stop_ep = epoch_rows[-1]["epoch"]
    epochs_after_best = stop_ep - best_ep

    best_row = next(r for r in epoch_rows if r["epoch"] == best_ep)
    last_row = epoch_rows[-1]

    # Gaps at best epoch
    acc_gap_at_best = (
        (best_row["train_acc"] - best_row["val_acc"])
        if best_row.get("train_acc") is not None and best_row.get("val_acc") is not None
        else None
    )
    f1_gap_at_best = (
        (best_row["train_macro_f1"] - best_row["val_macro_f1"])
        if best_row.get("train_macro_f1") is not None and best_row.get("val_macro_f1") is not None
        else None
    )
    loss_gap_at_best = (
        (best_row["val_loss"] - best_row["train_loss"])
        if best_row.get("val_loss") is not None and best_row.get("train_loss") is not None
        else None
    )

    # Validate against checkpoint metadata
    meta_issues: list[str] = []
    if checkpoint_meta_path.exists():
        try:
            meta = json.loads(checkpoint_meta_path.read_text(encoding="utf-8"))
            saved_best = meta.get("best_epoch")
            if saved_best is not None and int(saved_best) != best_ep:
                meta_issues.append(
                    f"History-derived best_epoch={best_ep} differs from "
                    f"checkpoint_metadata best_epoch={saved_best}"
                )
        except (OSError, json.JSONDecodeError, ValueError) as e:
            meta_issues.append(f"Could not read checkpoint_metadata.json: {e}")

    result.update({
        "status": "ok",
        "n_epochs": stop_ep,
        "best_epoch": best_ep,
        "stop_epoch": stop_ep,
        "epochs_after_best": epochs_after_best,
        "val_acc_at_best": best_row.get("val_acc"),
        "val_macro_f1_at_best": best_row.get("val_macro_f1"),
        "val_loss_at_best": best_row.get("val_loss"),
        "train_acc_at_best": best_row.get("train_acc"),
        "train_macro_f1_at_best": best_row.get("train_macro_f1"),
        "train_loss_at_best": best_row.get("train_loss"),
        "val_acc_at_stop": last_row.get("val_acc"),
        "val_macro_f1_at_stop": last_row.get("val_macro_f1"),
        "val_loss_at_stop": last_row.get("val_loss"),
        "acc_gap_at_best": acc_gap_at_best,
        "f1_gap_at_best": f1_gap_at_best,
        "loss_gap_at_best": loss_gap_at_best,
        "history_issues": issues,
        "meta_issues": meta_issues,
    })
    return result


# ---------------------------------------------------------------------------
# Trial loading
# ---------------------------------------------------------------------------

def _load_trials(tuning_dir: Path) -> list[dict[str, Any]]:
    """Load canonical all_trials.csv; reconcile counts against sweep_summary.csv."""
    canonical_path = tuning_dir / "all_trials.csv"
    sweep_path = tuning_dir / "sweep_summary.csv"
    issues: list[str] = []

    if not canonical_path.exists():
        raise FileNotFoundError(
            f"Canonical trial registry not found: {canonical_path}\n"
            "Run: python pipeline.py --stage tune"
        )

    with canonical_path.open("r", encoding="utf-8-sig", newline="") as f:
        canonical_rows = list(csv.DictReader(f))

    # Check for duplicate trial IDs (trial_id field if present)
    if "trial_id" in (canonical_rows[0] if canonical_rows else {}):
        ids = [r["trial_id"] for r in canonical_rows if r.get("trial_id")]
        duplicates = [tid for tid in set(ids) if ids.count(tid) > 1]
        if duplicates:
            issues.append(f"Duplicate trial_ids in canonical trials: {duplicates}")

    # Count per architecture
    arch_counts_canonical: dict[str, int] = {}
    for r in canonical_rows:
        a = r.get("architecture", "")
        arch_counts_canonical[a] = arch_counts_canonical.get(a, 0) + 1

    # Reconcile with sweep_summary.csv (may have additional attempt entries)
    if sweep_path.exists():
        with sweep_path.open("r", encoding="utf-8-sig", newline="") as f:
            sweep_rows = list(csv.DictReader(f))
        arch_counts_sweep: dict[str, int] = {}
        for r in sweep_rows:
            a = r.get("architecture", "")
            arch_counts_sweep[a] = arch_counts_sweep.get(a, 0) + 1
        for arch in set(arch_counts_canonical) | set(arch_counts_sweep):
            c = arch_counts_canonical.get(arch, 0)
            s = arch_counts_sweep.get(arch, 0)
            if s > c:
                issues.append(
                    f"{arch}: sweep_summary has {s} rows vs canonical {c}; "
                    "extra rows may be additional attempts not in canonical registry"
                )
    if issues:
        # Not fatal but report
        print(f"[diagnostics] Trial reconciliation notes: {issues}")

    for r in canonical_rows:
        r["_source"] = "canonical"
    return canonical_rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_best_vs_stop(dynamics: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    archs = list(dict.fromkeys(r["architecture"] for r in dynamics if r.get("status") == "ok"))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(archs)))
    arch_color = dict(zip(archs, colors))

    for ax, metric, ylabel in [
        (axes[0], "best_epoch", "Best epoch"),
        (axes[1], "stop_epoch", "Stop epoch"),
    ]:
        for arch in archs:
            arch_rows = [r for r in dynamics if r["architecture"] == arch and r.get(metric) is not None]
            seeds = [r["seed"] for r in arch_rows]
            vals = [r[metric] for r in arch_rows]
            ax.scatter(seeds, vals, color=arch_color[arch], label=arch, s=60, zorder=3)
        ax.set_xlabel("Seed")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " by seed")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    fig.suptitle("Best and stop epoch per run", y=1.01)
    fig.tight_layout()
    path = out_dir / "best_vs_stop_epoch.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_gaps(dynamics: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gap_keys = ["acc_gap_at_best", "f1_gap_at_best", "loss_gap_at_best"]
    gap_labels = [
        "Accuracy gap (train−val)",
        "Macro F1 gap (train−val)",
        "Loss gap (val−train)",
    ]

    archs = list(dict.fromkeys(r["architecture"] for r in dynamics if r.get("status") == "ok"))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(archs)))
    arch_color = dict(zip(archs, colors))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key, label in zip(axes, gap_keys, gap_labels):
        for arch in archs:
            arch_rows = [r for r in dynamics if r["architecture"] == arch
                         and r.get(key) is not None]
            seeds = [r["seed"] for r in arch_rows]
            vals = [r[key] for r in arch_rows]
            ax.scatter(seeds, vals, color=arch_color[arch], label=arch, s=60, zorder=3)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("Seed")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.25)

    note = ("Gaps use best-epoch train vs val. Train metrics include augmentation/param updates; "
            "validation uses eval mode. Gap is diagnostic, not a threshold for overfitting.")
    fig.text(0.5, -0.04, note, ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    path = out_dir / "training_gaps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_tuning_scatter(trial_rows: list[dict[str, Any]], out_dir: Path) -> None:
    """LR and weight decay scatter vs validation Macro F1, colour by optimizer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = [r for r in trial_rows if r.get("status") == "completed"
                 and _safe_float(r.get("best_val_macro_f1")) is not None
                 and _safe_float(r.get("learning_rate")) is not None]
    if not completed:
        return

    opt_colors = {"adamw": "#1565C0", "adam": "#2E7D32", "sgd": "#B71C1C"}
    archs = sorted(set(r.get("architecture", "?") for r in completed))

    fig, axes = plt.subplots(len(archs), 2, figsize=(10, 3.5 * len(archs)))
    if len(archs) == 1:
        axes = [axes]

    for i, arch in enumerate(archs):
        arch_trials = [r for r in completed if r.get("architecture") == arch]
        lrs = [float(r["learning_rate"]) for r in arch_trials]
        wds = [float(r["weight_decay"]) for r in arch_trials]
        f1s = [float(r["best_val_macro_f1"]) for r in arch_trials]
        opts = [str(r.get("optimizer", "")).lower() for r in arch_trials]
        colors = [opt_colors.get(o, "#666") for o in opts]

        ax_lr, ax_wd = axes[i][0], axes[i][1]
        for opt in ["adamw", "adam", "sgd"]:
            mask = [j for j, o in enumerate(opts) if o == opt]
            if mask:
                ax_lr.scatter([lrs[j] for j in mask], [f1s[j] for j in mask],
                               color=opt_colors[opt], label=f"{opt} (n={len(mask)})", s=50, alpha=0.75)
                ax_wd.scatter([wds[j] for j in mask], [f1s[j] for j in mask],
                               color=opt_colors[opt], label=f"{opt} (n={len(mask)})", s=50, alpha=0.75)
        ax_lr.set_xscale("log"); ax_wd.set_xscale("log")
        ax_lr.set_xlabel("Learning rate (log)")
        ax_lr.set_ylabel("Val Macro F1")
        ax_lr.set_title(f"{arch} — LR vs Val Macro F1")
        ax_lr.legend(fontsize=7); ax_lr.grid(alpha=0.25)
        ax_wd.set_xlabel("Weight decay (log)")
        ax_wd.set_ylabel("Val Macro F1")
        ax_wd.set_title(f"{arch} — WD vs Val Macro F1")
        ax_wd.legend(fontsize=7); ax_wd.grid(alpha=0.25)

    note = ("No causality implied: random budget differs per optimizer group; "
            "parameters are jointly sampled.")
    fig.text(0.5, -0.01, note, ha="center", fontsize=7, style="italic")
    fig.tight_layout()
    path = out_dir / "tuning_scatter.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_diagnostics_analysis(base_out_dir: Path, out_dir: Path,
                              selection: dict[str, Any]) -> dict[str, Any]:
    """Compute training dynamics and tuning analysis from cached artifacts."""
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    final_dir = base_out_dir / "final"
    tuning_dir = base_out_dir / "tuning"
    architectures = selection.get("architectures", [])
    seeds = [int(s) for s in selection.get("seeds", [])]

    # --- Training dynamics ---
    dynamics: list[dict[str, Any]] = []
    checkpoint_validation_rows: list[dict[str, Any]] = []
    issues: list[str] = []

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
        run_dir = config_path.parent
        dyn = _extract_dynamics(arch, int(seed), run_dir)
        dynamics.append(dyn)

        if dyn.get("history_issues"):
            issues.extend(dyn["history_issues"])
        if dyn.get("meta_issues"):
            issues.extend(dyn["meta_issues"])

        # Checkpoint validation row
        checkpoint_validation_rows.append({
            "architecture": arch, "seed": seed,
            "history_derived_best_epoch": dyn.get("best_epoch"),
            "history_n_epochs": dyn.get("n_epochs"),
            "checkpoint_meta_match": not bool(dyn.get("meta_issues")),
            "issues": "; ".join(dyn.get("meta_issues", [])),
        })

    # Architecture dynamics summary
    arch_dyn_summary: list[dict[str, Any]] = []
    for arch in architectures:
        subset = [d for d in dynamics if d["architecture"] == arch and d.get("status") == "ok"]
        summary: dict[str, Any] = {"architecture": arch, "completed_runs": len(subset)}
        for key in ["best_epoch", "stop_epoch", "epochs_after_best",
                    "acc_gap_at_best", "f1_gap_at_best", "loss_gap_at_best"]:
            vals = [d[key] for d in subset if d.get(key) is not None]
            summary[key + "_mean"] = float(np.mean(vals)) if vals else None
            summary[key + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) >= 2 else None
        arch_dyn_summary.append(summary)

    # --- Tuning trials ---
    trial_rows: list[dict[str, Any]] = []
    try:
        trial_rows = _load_trials(tuning_dir)
    except FileNotFoundError as e:
        issues.append(str(e))

    # Separate failed trials
    completed_trials = [r for r in trial_rows if r.get("status") == "completed"]
    failed_trials = [r for r in trial_rows if r.get("status") != "completed"]

    trial_inventory_rows: list[dict[str, Any]] = []
    for arch in architectures:
        arch_trials = [r for r in trial_rows if r.get("architecture") == arch]
        arch_complete = [r for r in arch_trials if r.get("status") == "completed"]
        trial_inventory_rows.append({
            "architecture": arch,
            "n_canonical_trials": len(arch_trials),
            "n_completed": len(arch_complete),
            "n_failed": len(arch_trials) - len(arch_complete),
        })

    # Figures
    if dynamics:
        _plot_best_vs_stop(dynamics, diag_dir)
        _plot_gaps(dynamics, diag_dir)
    if trial_rows:
        _plot_tuning_scatter(trial_rows, diag_dir)

    # Save outputs
    save_csv(diag_dir / "run_dynamics.csv", dynamics)
    save_csv(diag_dir / "architecture_dynamics_summary.csv", arch_dyn_summary)
    save_csv(diag_dir / "trial_inventory.csv", trial_inventory_rows)
    save_csv(diag_dir / "checkpoint_validation.csv", checkpoint_validation_rows)
    save_json(diag_dir / "issues.json", {"issues": issues})

    return {
        "dynamics": dynamics,
        "arch_dyn_summary": arch_dyn_summary,
        "trial_rows": trial_rows,
        "issues": issues,
        "out_dir": str(diag_dir),
    }

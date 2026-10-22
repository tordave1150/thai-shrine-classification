"""Extension report generator.

Summarises whatever artifacts actually exist in outputs_extensions_v1/.
Never starts training. Reports missing/incomplete outputs explicitly.
Generates slide-ready figures for comparison results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from shrine_classifier.analysis import save_json, save_csv
from shrine_classifier.data import CLASSES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _plot_condition_summary(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    """Bar chart of mean test Macro F1 per condition with seed dots and SD error bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not summary_rows:
        return

    conditions = [r["condition"] for r in summary_rows if r.get("condition")]
    means = []
    sds = []
    for r in summary_rows:
        m = r.get("test_macro_f1_mean")
        s = r.get("test_macro_f1_sd")
        means.append(float(m) if m is not None else 0.0)
        sds.append(float(s) if s is not None else 0.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=sds, capsize=4, color="#1565C0", alpha=0.75,
                  error_kw={"elinewidth": 1.5, "ecolor": "#333"})

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Mean Test Macro F1 (± sample SD, n=5)")
    ax.set_title("Extension conditions — Test Macro F1\n"
                 "[EXPLORATORY: test set already inspected during final evaluation]")
    ax.set_ylim(max(0, min(means) - 0.1) if means else 0, 1.0)
    ax.grid(alpha=0.25, axis="y")

    note = ("Exploratory comparison on a test split inspected during the primary final evaluation. "
            "n=5 reflects seed variation on one fixed split only.")
    fig.text(0.5, -0.06, note, ha="center", fontsize=7, style="italic", wrap=True)
    fig.tight_layout()
    path = out_dir / "condition_comparison_macro_f1.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_condition_accuracy(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not summary_rows:
        return

    conditions = [r["condition"] for r in summary_rows if r.get("condition")]
    means = [float(r["test_acc_mean"]) if r.get("test_acc_mean") is not None else 0.0 for r in summary_rows]
    sds = [float(r["test_acc_sd"]) if r.get("test_acc_sd") is not None else 0.0 for r in summary_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(conditions))
    ax.bar(x, means, yerr=sds, capsize=4, color="#2E7D32", alpha=0.75,
           error_kw={"elinewidth": 1.5, "ecolor": "#333"})
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylabel("Mean Test Accuracy (± sample SD, n=5)")
    ax.set_title("Extension conditions — Test Accuracy\n[EXPLORATORY]")
    ax.set_ylim(max(0, min(means) - 0.1) if means else 0, 1.0)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    path = out_dir / "condition_comparison_accuracy.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run_report(out_dir: Path, base_out_dir: Path) -> dict[str, Any]:
    """Summarise all extension artifacts and generate slide-ready figures."""
    report_dir = out_dir / "report_assets"
    report_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, str] = {}

    # --- Confidence analysis ---
    conf_run_metrics = _read_csv(out_dir / "confidence" / "run_metrics.csv")
    conf_arch_summary = _read_csv(out_dir / "confidence" / "architecture_summary.csv")
    conf_bin_metrics = _read_csv(out_dir / "confidence" / "bin_metrics.csv")
    conf_hce = _read_csv(out_dir / "confidence" / "high_confidence_errors.csv")
    status["confidence_run_metrics"] = "present" if conf_run_metrics else "missing"
    status["confidence_arch_summary"] = "present" if conf_arch_summary else "missing"

    # --- Diagnostics analysis ---
    diag_dynamics = _read_csv(out_dir / "diagnostics" / "run_dynamics.csv")
    diag_arch_summary = _read_csv(out_dir / "diagnostics" / "architecture_dynamics_summary.csv")
    diag_trials = _read_csv(out_dir / "diagnostics" / "trial_inventory.csv")
    status["diagnostics_run_dynamics"] = "present" if diag_dynamics else "missing"
    status["diagnostics_arch_summary"] = "present" if diag_arch_summary else "missing"

    # --- Comparisons (extension experiments) ---
    comp_summary = _read_csv(out_dir / "comparisons" / "condition_summary.csv")
    comp_tests = _read_csv(out_dir / "comparisons" / "statistical_tests.csv")
    comp_per_class = _read_csv(out_dir / "comparisons" / "per_class_deltas.csv")
    status["comparisons_summary"] = "present" if comp_summary else "missing"
    status["comparisons_statistical_tests"] = "present" if comp_tests else "missing"

    # --- Protocol ---
    protocol = _read_json(out_dir / "protocol.json")
    status["protocol"] = "present" if protocol else "missing"

    # --- Figures for presentation ---
    if comp_summary:
        _plot_condition_summary(comp_summary, report_dir)
        _plot_condition_accuracy(comp_summary, report_dir)
        status["figure_condition_macro_f1"] = "generated"
        status["figure_condition_accuracy"] = "generated"
    else:
        status["figure_condition_macro_f1"] = "skipped_no_data"
        status["figure_condition_accuracy"] = "skipped_no_data"

    # --- README ---
    readme_lines = [
        "# Extension Report Assets",
        "",
        "Generated by `pipeline_extensions.py --stage report`.",
        "This is supplementary to the primary report; it does not overwrite primary outputs.",
        "",
        "## Artifact status",
        "",
    ]
    for key, val in sorted(status.items()):
        icon = "✅" if val in ("present", "generated") else "❌"
        readme_lines.append(f"- {icon} `{key}`: {val}")
    readme_lines += [
        "",
        "## Key outputs",
        "",
        "| File | Description |",
        "|---|---|",
        "| `confidence/run_metrics.csv` | Per-run ECE, correct/wrong confidence stats |",
        "| `confidence/architecture_summary.csv` | Per-arch ECE mean ± SD |",
        "| `confidence/reliability_<arch>.png` | Reliability diagram (representative seed) |",
        "| `confidence/confidence_hist_<arch>.png` | Confidence histogram |",
        "| `diagnostics/run_dynamics.csv` | Per-run best/stop epoch, gaps |",
        "| `diagnostics/training_gaps.png` | Acc/F1/loss gap figures |",
        "| `diagnostics/tuning_scatter.png` | LR/WD vs val Macro F1 |",
        "| `comparisons/condition_summary.csv` | Mean ± SD per condition |",
        "| `comparisons/statistical_tests.csv` | Paired Holm tests |",
        "| `report_assets/condition_comparison_*.png` | Slide-ready comparison bar charts |",
        "",
        "## Caveats",
        "",
        "- Extension experiments (ablation / fine-tuning modes) use a test set that was "
          "already inspected during the primary final evaluation. These are **exploratory** results.",
        "- n=5 seeds on one fixed split reflects seed variation only, not dataset uncertainty.",
        "- Holm family for extension: 4 variants × 2 metrics = 8 comparisons. "
          "Separate from the primary 8-comparison family.",
    ]
    (report_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    # Summary JSON
    summary = {
        "status": status,
        "n_confidence_runs": len(conf_run_metrics),
        "n_diagnostic_dynamics": len(diag_dynamics),
        "n_comp_conditions": len(comp_summary),
        "n_stat_tests": len(comp_tests),
    }
    save_json(out_dir / "extension_summary.json", summary)

    # Also write Markdown extension summary
    md_lines = [
        "# Extension summary",
        "",
        "## Artifact status",
        "",
    ]
    for key, val in sorted(status.items()):
        md_lines.append(f"- `{key}`: **{val}**")
    md_lines += [
        "",
        "## Confidence analysis (Item 1)",
        f"- Runs analysed: {len(conf_run_metrics)}",
        f"- Architectures with ECE summary: {len(conf_arch_summary)}",
        "",
        "## Training dynamics (Item 2)",
        f"- Final runs analysed: {len(diag_dynamics)}",
        f"- Trial inventory entries: {len(diag_trials)}",
        "",
        "## Extension experiments (Items 3–4)",
        f"- Conditions with results: {len(comp_summary)}",
        f"- Paired tests recorded: {len(comp_tests)}",
        "",
        "> This summary was generated from existing artifacts only. "
        "Missing entries above indicate stages not yet completed.",
    ]
    (out_dir / "extension_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {"status": status, "out_dir": str(report_dir)}

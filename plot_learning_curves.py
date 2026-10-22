"""
plot_learning_curves.py
=======================
Plot mean ± 1 SD learning curves (loss and accuracy) for every
architecture × mode pair in outputs_modes_v3_td.

Each pair's 5 seeds are aggregated epoch-by-epoch:
  - Solid/dashed line  = mean across seeds
  - Shaded band        = ±1 SD across seeds
  - Dotted vertical    = mean best epoch (from run_metrics.json)

Outputs
-------
  outputs_modes_v3_td/final/curves/<arch>/<mode>/mean_sd_curves.png
      one figure per pair (loss + acc subplots)

  outputs_modes_v3_td/final/curves/<arch>/mean_sd_grid.png
      3-mode side-by-side grid per architecture (mirrors reference Fig 7.2)

  outputs_modes_v3_td/final/curves/all_pairs_grid.png
      single large overview of all 15 pairs

Usage
-----
  python plot_learning_curves.py
  python plot_learning_curves.py --out-dir outputs_modes_v3_td --metric loss
  python plot_learning_curves.py --arch efficientnet_b0 --mode full frozen
  python plot_learning_curves.py --no-per-pair --no-overview
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── canonical orders (AGENTS.md) ────────────────────────────────────────────
ARCHITECTURES = [
    "resnet50", "efficientnet_b0", "mobilenet_v3_large",
    "densenet121", "convnext_tiny",
]
MODES = ["full", "frozen", "partial"]
SEEDS = [42, 123, 456, 789, 1024]

ARCH_LABELS = {
    "resnet50":           "ResNet-50",
    "efficientnet_b0":    "EfficientNet-B0",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "densenet121":        "DenseNet-121",
    "convnext_tiny":      "ConvNeXt-Tiny",
}
MODE_LABELS = {
    "full":    "Full fine-tuning",
    "frozen":  "Frozen backbone",
    "partial": "Partial (last stage)",
}

# ── visual style ─────────────────────────────────────────────────────────────
TRAIN_COLOR = "#4878CF"
VAL_COLOR   = "#E07B39"
BEST_COLOR  = "#666666"
BAND_ALPHA  = 0.20

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── data helpers ─────────────────────────────────────────────────────────────

def find_run_dirs(base: Path, arch: str, mode: str) -> list:
    pair_dir = base / "final" / arch / mode
    if not pair_dir.exists():
        return []
    return sorted(d for d in pair_dir.iterdir() if d.is_dir())


def load_history(run_dir: Path):
    csv = run_dir / "history.csv"
    if not csv.exists():
        return None
    try:
        df = pd.read_csv(csv)
        required = {"epoch", "train_loss", "val_loss", "train_acc", "val_acc"}
        if not required.issubset(df.columns):
            print(f"    [warn] missing columns in {csv}")
            return None
        # Drop summary/test rows appended at the end (they have NaN train_loss)
        df = df[df["train_loss"].notna()].copy()
        # Deduplicate by epoch, keeping the first (actual training) row
        df = df.drop_duplicates(subset="epoch", keep="first")
        return df.set_index("epoch")
    except Exception as exc:
        print(f"    [warn] cannot read {csv}: {exc}")
        return None


def load_best_epoch(run_dir: Path):
    p = run_dir / "run_metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())["best_epoch"]
    except Exception:
        return None


def aggregate(dfs: list, col: str):
    """
    Align per-seed histories by epoch (NaN-pad shorter runs).
    Returns (epochs_1d, mean_1d, sd_1d).
    """
    if not dfs:
        return np.array([]), np.array([]), np.array([])

    max_ep = max(df.index.max() for df in dfs)
    epochs = np.arange(1, int(max_ep) + 1)
    arr = np.full((len(dfs), len(epochs)), np.nan)
    for i, df in enumerate(dfs):
        vals = df[col].reindex(epochs).values
        arr[i, :len(vals)] = vals

    mean = np.nanmean(arr, axis=0)
    sd   = np.nanstd(arr, axis=0, ddof=1)
    sd   = np.where(np.isnan(sd), 0.0, sd)   # single-seed → 0 not NaN
    return epochs, mean, sd


# ── low-level drawing ────────────────────────────────────────────────────────

def _draw(
    ax, epochs,
    tr_mean, tr_sd,
    vl_mean, vl_sd,
    mean_best_epoch,
    ylabel, title,
    show_legend=True,
    show_xlabel=True,
):
    h1, = ax.plot(epochs, tr_mean, color=TRAIN_COLOR, linestyle="--",
                  linewidth=1.6, label="Train (mean ± 1 SD)")
    ax.fill_between(epochs, tr_mean - tr_sd, tr_mean + tr_sd,
                    color=TRAIN_COLOR, alpha=BAND_ALPHA)

    h2, = ax.plot(epochs, vl_mean, color=VAL_COLOR, linestyle="-",
                  linewidth=1.6, label="Validation (mean ± 1 SD)")
    ax.fill_between(epochs, vl_mean - vl_sd, vl_mean + vl_sd,
                    color=VAL_COLOR, alpha=BAND_ALPHA)

    h3 = None
    if mean_best_epoch is not None:
        h3 = ax.axvline(mean_best_epoch, color=BEST_COLOR, linestyle=":",
                        linewidth=1.2, label="Mean best epoch")

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_ylabel(ylabel, fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Epoch", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55)
    ax.set_xlim(left=1)

    if show_legend:
        handles = [h1, h2] + ([h3] if h3 else [])
        ax.legend(handles=handles, fontsize=6.5, loc="best", framealpha=0.85)

    return [h1, h2] + ([h3] if h3 else [])


def _collect(base, arch, mode):
    """Return (dfs, mean_best_epoch_or_None)."""
    run_dirs    = find_run_dirs(base, arch, mode)
    dfs         = [df for rd in run_dirs if (df := load_history(rd)) is not None]
    best_epochs = [be for rd in run_dirs if (be := load_best_epoch(rd)) is not None]
    mean_be     = float(np.mean(best_epochs)) if best_epochs else None
    return dfs, mean_be


# ── figure 1: per-pair ───────────────────────────────────────────────────────

def plot_pair(base: Path, arch: str, mode: str, save_dir: Path, metric: str = "all"):
    """Loss / accuracy (/ F1) subplots for one arch × mode pair."""
    dfs, mean_be = _collect(base, arch, mode)
    if not dfs:
        print(f"    [skip] {arch}/{mode} — no valid history.csv")
        return None

    n = len(dfs)
    has_f1 = "train_macro_f1" in dfs[0].columns

    rows_to_plot = []
    if metric in ("all", "loss"):
        rows_to_plot.append("loss")
    if metric in ("all", "acc"):
        rows_to_plot.append("acc")
    if metric in ("all", "f1") and has_f1:
        rows_to_plot.append("f1")

    fig, axes = plt.subplots(len(rows_to_plot), 1,
                             figsize=(6, 3 * len(rows_to_plot)), squeeze=False)
    fig.suptitle(
        f"Learning curves (mean ± SD, n={n} seeds)\n"
        f"{ARCH_LABELS.get(arch, arch)} — {MODE_LABELS.get(mode, mode)}",
        fontsize=10, fontweight="bold", y=1.01,
    )

    for r, m in enumerate(rows_to_plot):
        ax = axes[r, 0]
        is_last = (r == len(rows_to_plot) - 1)
        label = chr(97 + r)
        if m == "loss":
            ep, tr_m, tr_s = aggregate(dfs, "train_loss")
            _,  vl_m, vl_s = aggregate(dfs, "val_loss")
            _draw(ax, ep, tr_m, tr_s, vl_m, vl_s, mean_be,
                  ylabel="Weighted CE loss", title=f"({label}) Loss",
                  show_legend=(r == 0), show_xlabel=is_last)
        elif m == "acc":
            ep, tr_m, tr_s = aggregate(dfs, "train_acc")
            _,  vl_m, vl_s = aggregate(dfs, "val_acc")
            _draw(ax, ep, tr_m, tr_s, vl_m, vl_s, mean_be,
                  ylabel="Accuracy", title=f"({label}) Accuracy",
                  show_legend=(r == 0), show_xlabel=is_last)
            ax.set_ylim(0, 1.05)
        elif m == "f1":
            ep, tr_m, tr_s = aggregate(dfs, "train_macro_f1")
            _,  vl_m, vl_s = aggregate(dfs, "val_macro_f1")
            _draw(ax, ep, tr_m, tr_s, vl_m, vl_s, mean_be,
                  ylabel="Macro F1", title=f"({label}) Macro F1",
                  show_legend=(r == 0), show_xlabel=is_last)
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "mean_sd_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {out}")
    return out


# ── figure 2: per-architecture 3-mode grid (mirrors reference Fig 7.2) ──────

def plot_per_arch_grid(base: Path, arch: str, save_path: Path):
    """
    2 rows × 3 cols
      rows:    loss (top), accuracy (bottom)
      columns: full, frozen, partial
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    arch_label = ARCH_LABELS.get(arch, arch)

    fig.suptitle(
        f"Learning curves (mean ± SD over seeds) — {arch_label}",
        fontsize=12, fontweight="bold",
    )

    legend_handles = None

    for m_idx, mode in enumerate(MODES):
        dfs, mean_be = _collect(base, arch, mode)
        n = len(dfs)
        ml = MODE_LABELS.get(mode, mode)

        for row, metric in enumerate(["loss", "acc"]):
            ax = axes[row, m_idx]
            # subplot label: (a)…(f) column-major order to match reference
            col_letter = chr(97 + row * 3 + m_idx)
            sub_title = f"({col_letter}) {ml} — {metric}"
            if n:
                sub_title += f"  [n={n}]"

            if not dfs:
                ax.set_title(sub_title + "\n(no data)", fontsize=8.5)
                continue

            col_tr = "train_loss" if metric == "loss" else "train_acc"
            col_vl = "val_loss"   if metric == "loss" else "val_acc"
            ylabel = "Weighted CE loss" if metric == "loss" else "Accuracy"

            ep, tr_m, tr_s = aggregate(dfs, col_tr)
            _,  vl_m, vl_s = aggregate(dfs, col_vl)

            handles = _draw(
                ax, ep, tr_m, tr_s, vl_m, vl_s, mean_be,
                ylabel=ylabel, title=sub_title,
                show_legend=False,
                show_xlabel=(row == 1),
            )
            if legend_handles is None:
                legend_handles = handles

            if metric == "acc":
                ax.set_ylim(0, 1.05)

    # shared legend below title
    if legend_handles:
        labels = [h.get_label() for h in legend_handles]
        fig.legend(legend_handles, labels,
                   loc="upper center", ncol=len(legend_handles),
                   fontsize=8, bbox_to_anchor=(0.5, 0.975), framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {save_path}")


# ── figure 3: large overview (all architectures × all modes) ────────────────

def plot_overview_grid(base: Path, save_path: Path):
    """
    Rows = architecture × metric (loss then acc), Columns = mode.
    5 architectures × 2 metrics = 10 rows; 3 mode columns.
    """
    n_arch  = len(ARCHITECTURES)
    n_rows  = n_arch * 2    # loss + acc per arch
    n_cols  = len(MODES)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    fig.suptitle(
        "Learning curves — mean ± 1 SD over seeds  (all architectures × modes)",
        fontsize=14, fontweight="bold", y=1.003,
    )

    legend_handles = None

    for a_idx, arch in enumerate(ARCHITECTURES):
        arch_label = ARCH_LABELS.get(arch, arch)
        for m_idx, mode in enumerate(MODES):
            dfs, mean_be = _collect(base, arch, mode)
            ml = MODE_LABELS.get(mode, mode)

            for row_offset, metric in enumerate(["loss", "acc"]):
                row = a_idx * 2 + row_offset
                ax  = axes[row, m_idx]

                col_tr = "train_loss" if metric == "loss" else "train_acc"
                col_vl = "val_loss"   if metric == "loss" else "val_acc"
                ylabel = "CE loss" if metric == "loss" else "Accuracy"
                title  = f"{arch_label}\n{ml} — {metric}"

                if not dfs:
                    ax.set_title(title + "\n(no data)", fontsize=7)
                    continue

                ep, tr_m, tr_s = aggregate(dfs, col_tr)
                _,  vl_m, vl_s = aggregate(dfs, col_vl)

                handles = _draw(
                    ax, ep, tr_m, tr_s, vl_m, vl_s, mean_be,
                    ylabel=ylabel, title=title,
                    show_legend=False,
                    show_xlabel=(row_offset == 1),
                )
                if legend_handles is None:
                    legend_handles = handles

                if metric == "acc":
                    ax.set_ylim(0, 1.05)

    if legend_handles:
        labels = [h.get_label() for h in legend_handles]
        fig.legend(legend_handles, labels,
                   loc="upper center", ncol=len(legend_handles),
                   fontsize=9, bbox_to_anchor=(0.5, 1.005), framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved overview] {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot mean ± SD learning curves from history.csv logs.")
    p.add_argument("--out-dir", default="outputs_modes_v3_td",
                   help="Root output dir  (default: outputs_modes_v3_td)")
    p.add_argument("--metric", choices=["loss", "acc", "f1", "all"], default="all",
                   help="Metric(s) for per-pair figures  (default: all)")
    p.add_argument("--arch", nargs="+", default=None,
                   help="Limit to specific architecture(s)  (default: all 5)")
    p.add_argument("--mode", nargs="+", default=None,
                   help="Limit to specific mode(s)  (default: all 3)")
    p.add_argument("--no-per-pair", action="store_true",
                   help="Skip per-pair figures")
    p.add_argument("--no-per-arch", action="store_true",
                   help="Skip per-architecture 3-mode grid figures")
    p.add_argument("--no-overview", action="store_true",
                   help="Skip the large all-pairs overview grid")
    return p.parse_args()


def main():
    args  = parse_args()
    base  = Path(args.out_dir)
    archs = args.arch or ARCHITECTURES
    modes = args.mode or MODES
    curves_root = base / "final" / "curves"

    print(f"Output root  : {base.resolve()}")
    print(f"Architectures: {archs}")
    print(f"Modes        : {modes}")
    print()

    # 1. Per-pair
    if not args.no_per_pair:
        print("=== Per-pair figures ===")
        for arch in archs:
            for mode in modes:
                print(f"  {arch} / {mode}")
                plot_pair(base, arch, mode,
                          save_dir=curves_root / arch / mode,
                          metric=args.metric)
        print()

    # 2. Per-architecture 3-mode grids
    if not args.no_per_arch:
        print("=== Per-architecture 3-mode grids ===")
        for arch in archs:
            out = curves_root / arch / "mean_sd_grid.png"
            print(f"  {arch}")
            plot_per_arch_grid(base, arch, save_path=out)
        print()

    # 3. Overview grid
    if not args.no_overview:
        print("=== Overview grid (all pairs) ===")
        plot_overview_grid(base, save_path=curves_root / "all_pairs_grid.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

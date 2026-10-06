"""Shared chart style and drawing functions for the scripts, notebook and slides.

All charts read saved outputs in results/; nothing here trains or re-tunes a model.
Fonts: bundled Open Sans / Montserrat (SIL OFL, assets/fonts/). If those files are
missing, matplotlib falls back to its default sans-serif, so the charts still build
on Windows, macOS and Linux.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"

# Modern & clean palette shared with graphs.html and the slides.
INK = "#0F172A"        # titles / primary text
TEXT = "#334155"       # body text
MUTED = "#64748B"      # secondary text
GRID = "#E2E8F0"       # recessive grid / axes
SURFACE = "#FFFFFF"
ACCENT = "#EA580C"     # eruption / highlight (volcanic orange)
PRIMARY = "#1D4ED8"    # selected model / primary series
NEUTRAL = "#94A3B8"    # baselines and chance lines

# Fixed categorical order (validated: CVD-safe adjacent pairs, labels always shown).
FAMILY_COLORS = {"baseline": NEUTRAL, "logistic": "#2A78D6", "forest": "#EB6834",
                 "kmeans": "#1BAF7A", "mlp": "#EDA100"}
FAMILY_STYLES = {"baseline": ":", "logistic": "-", "forest": "--", "kmeans": "-", "mlp": "-."}
CLASS_LABELS = {"baseline": "Prior baseline", "logistic": "Logistic regression",
                "forest": "Random forest", "kmeans": "Class-conditional K-means",
                "mlp": "Neural network (MLP)"}
REG_LABELS = {"baseline": "Mean baseline", "forest": "Random forest",
              "kmeans": "Joint K-means", "mlp": "Neural network (MLP)"}
CLASS_FAMILIES = ["baseline", "logistic", "forest", "kmeans", "mlp"]
REG_FAMILIES = ["baseline", "forest", "kmeans", "mlp"]


def use_style():
    """Register bundled fonts if present and apply the shared rcParams."""
    family = []
    for name in ("OpenSans-Regular", "OpenSans-SemiBold", "OpenSans-Bold",
                 "Montserrat-Regular", "Montserrat-SemiBold", "Montserrat-Bold"):
        path = FONT_DIR / f"{name}.ttf"
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Open Sans" in names:
        family.append("Open Sans")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": family + ["Calibri", "Arial", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 12.5, "axes.titleweight": "semibold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 12,
        "axes.labelsize": 10, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.fontsize": 8.5, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return "Montserrat" if "Montserrat" in names else plt.rcParams["font.sans-serif"][0]


def axis_style(ax, grid_axis="both"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def load_results(protocol="chronological"):
    import json
    folder = ROOT / "results" / protocol
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(folder / "test_predictions.csv")
    return metrics, pred


def draw_roc(ax, pred, metrics):
    y = pred["erupting"].astype(int).to_numpy()
    selected = metrics["selection"]["classification"]["family"]
    for family in CLASS_FAMILIES:
        score = pred[f"classification_{family}"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        star = " *" if family == selected else ""
        ax.plot(fpr, tpr, color=FAMILY_COLORS[family], linestyle=FAMILY_STYLES[family],
                linewidth=2.6 if family == selected else 1.8, drawstyle="steps-post",
                label=f"{CLASS_LABELS[family]}{star}  AUROC {auc:.3f}", zorder=3 if family == selected else 2)
    ax.plot([0, 1], [0, 1], color=INK, linestyle=(0, (4, 4)), linewidth=1, alpha=.6, label="Chance  0.500")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("Eruption classification: ROC on the chronological test period")
    ax.text(0.02, 0.98, f"n = {len(y)} earthquakes, {int(y.sum())} erupting  ·  * selected on development data",
            transform=ax.transAxes, va="top", color=MUTED, fontsize=8.5)
    ax.legend(loc="lower right", frameon=False)
    axis_style(ax)


def draw_confusion(ax, pred, metrics):
    y = pred["erupting"].astype(int).to_numpy()
    family = metrics["selection"]["classification"]["family"]
    config = metrics["selection"]["classification"]["config"]
    predicted = (pred[f"classification_{family}"].astype(float).to_numpy() >= .5).astype(int)
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#EFF6FF", PRIMARY])
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1], labels=["Repose", "Erupting"])
    ax.set_yticks([0, 1], labels=["Repose", "Erupting"])
    ax.set_xlabel("Predicted label (score ≥ 0.5)")
    ax.set_ylabel("Catalog label")
    ax.set_title(f"Confusion matrix: {CLASS_LABELS[family]} ({config})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=20, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * .55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return cm


def draw_rmse(ax, metrics):
    reg = metrics["metrics"]["regression"]
    selected = metrics["selection"]["regression"]["family"]
    values = [reg[f]["test"]["rmse_days"] for f in REG_FAMILIES]
    y = np.arange(len(REG_FAMILIES))
    colors = [PRIMARY if f == selected else NEUTRAL for f in REG_FAMILIES]
    bars = ax.barh(y, values, color=colors, height=.58)
    ax.set_yticks(y, labels=[REG_LABELS[f] + (" *" if f == selected else "") for f in REG_FAMILIES])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("Test RMSE in days (lower is better)")
    ax.set_title("Time to eruption: test error by model")
    for bar, f, value in zip(bars, REG_FAMILIES, values):
        r2 = reg[f]["test"]["r2"]
        ax.text(value + max(values) * .02, bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} d   (R² {r2:.3f})", va="center", color=INK, fontsize=9)
    ax.text(1, -0.16, "* selected on development data", transform=ax.transAxes, ha="right",
            color=MUTED, fontsize=8.5)
    axis_style(ax, grid_axis="x")


def draw_observed_predicted(ax, pred, metrics):
    selected = metrics["selection"]["regression"]["family"]
    observed = pred["tte_days"].astype(float).to_numpy()
    others = [f for f in ("forest",) if f != selected]
    upper = max(float(observed.max()),
                *(float(pred[f"regression_{f}"].max()) for f in [selected] + others)) * 1.05
    ax.plot([0, upper], [0, upper], color=INK, linestyle=(0, (4, 4)), linewidth=1, alpha=.6,
            label="Perfect prediction")
    for f in others:
        ax.scatter(observed, pred[f"regression_{f}"], s=22, alpha=.55, color=FAMILY_COLORS[f],
                   edgecolor="white", linewidth=.5, label=f"{REG_LABELS[f]} (candidate)")
    ax.scatter(observed, pred[f"regression_{selected}"], s=22, alpha=.7, color=PRIMARY,
               edgecolor="white", linewidth=.5, label=f"{REG_LABELS[selected]} (selected)")
    ax.set(xlim=(0, upper), ylim=(0, upper), xlabel="Observed days to next eruption",
           ylabel="Predicted days")
    ax.set_title("Time to eruption: observed vs predicted")
    ax.legend(loc="upper left", frameon=False)
    axis_style(ax)


def draw_protocols(ax, chrono, random):
    """Selected-model test metrics under each split (each split selects its own model)."""
    rows = []
    for name, m in (("Chronological", chrono), ("Random", random)):
        cf = m["selection"]["classification"]["family"]
        rf = m["selection"]["regression"]["family"]
        rows.append((name, m["metrics"]["classification"][cf]["test"]["auroc"],
                     m["metrics"]["classification"][cf]["test"]["kappa"],
                     m["metrics"]["regression"][rf]["test"]["r2"], cf, rf))
    metrics_names = ["AUROC", "Cohen's kappa", "Regression R²"]
    x = np.arange(len(metrics_names))
    w = .34
    for k, (name, auc, kap, r2, cf, rf) in enumerate(rows):
        vals = [auc, kap, r2]
        color = PRIMARY if name == "Chronological" else NEUTRAL
        bars = ax.bar(x + (k - .5) * w, vals, w * .92, color=color, label=f"{name} split")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (.02 if v >= 0 else -.02), f"{v:.3f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK, linewidth=.8)
    ax.plot([-.42, .42], [.5, .5], color=INK, linestyle=(0, (4, 4)), linewidth=.9, alpha=.6)
    ax.text(.46, .5, "chance", fontsize=8, color=MUTED, va="center")
    sel = "  ·  ".join(f"{n}: {CLASS_LABELS[cf]} / {REG_LABELS[rf]}" for n, _, _, _, cf, rf in rows)
    ax.text(0, -.13, "Dev-selected models (classifier / regressor) — " + sel, transform=ax.transAxes,
            fontsize=7.5, color=MUTED)
    ax.set_xticks(x, labels=metrics_names)
    ax.set_ylim(-.3, .85)
    ax.set_title("Same features, two evaluation splits")
    ax.legend(loc="upper right", frameon=False)
    axis_style(ax, grid_axis="y")


def draw_timeline(ax, prepared):
    """Weekly earthquake counts with eruptive labels and the chronological split windows."""
    d = prepared.copy()
    d["time"] = pd.to_datetime(d["time"])
    weekly = d.set_index("time").resample("7D").size()
    ax.bar(weekly.index, weekly.values, width=6, color=NEUTRAL, label="Earthquakes per week")
    erupt = d[d.erupting == 1]
    ax.scatter(erupt.time, np.full(len(erupt), -2.5), marker="|", s=40, color=ACCENT,
               label="Earthquake during eruptive episode")
    shades = {"train": "#DBEAFE", "dev": "#FEF3C7", "test": "#DCFCE7"}
    names = {"train": "Train", "dev": "Development", "test": "Test"}
    top = weekly.max() * 1.18
    for split in ("train", "dev", "test"):
        s = d[d.split == split].time
        ax.axvspan(s.min(), s.max(), color=shades[split], zorder=0)
        ax.text(s.min() + (s.max() - s.min()) / 2, top * .94, f"{names[split]}\n{len(s):,} quakes",
                ha="center", va="top", fontsize=8.5, color=INK)
    ax.set_ylim(-5, top)
    ax.set_ylabel("Earthquakes per week")
    ax.set_title("East Rift Zone earthquakes, 1983–1986, and the time-ordered split")
    ax.legend(loc="center left", bbox_to_anchor=(0, .62), frameon=False)
    axis_style(ax, grid_axis="y")


PUU_OO = (-155.1050, 19.3864)  # longitude, latitude used in the original RunMLTests.ipynb


def draw_map(ax, prepared):
    """Modeled East Rift Zone epicentres coloured by depth, with Pu'u 'O'o marked."""
    d = prepared
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("depth", ["#FDBA74", "#1D4ED8", "#0F172A"])
    sc = ax.scatter(d.longitude, d.latitude, c=d.depth, cmap=cmap, s=7, alpha=.75, linewidth=0, vmin=0, vmax=15)
    ax.scatter(*PUU_OO, marker="^", s=140, color=ACCENT, edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate("Pu'u 'Ō'ō", PUU_OO, xytext=(10, 10), textcoords="offset points", color=INK,
                fontsize=10, fontweight="bold")
    ax.set(xlabel="Longitude (°)", ylabel="Latitude (°)")
    ax.set_aspect(1 / np.cos(np.deg2rad(19.35)))
    ax.set_title(f"{len(d):,} modeled East Rift Zone earthquakes, 1983–1986")
    axis_style(ax)
    cb = plt.colorbar(sc, ax=ax, shrink=.7, pad=.02)
    cb.set_label("Depth (km)", color=TEXT)
    cb.outline.set_visible(False)
    cb.ax.invert_yaxis()
    cb.ax.tick_params(colors=MUTED, length=0)

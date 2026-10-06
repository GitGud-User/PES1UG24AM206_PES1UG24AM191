"""Create presentation charts from the saved results (no model is retrained here)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import plots  # noqa: E402

OUT = ROOT / "output" / "figures"


def save_one(filename, size, draw, *args):
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    draw(ax, *args)
    fig.savefig(OUT / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create():
    OUT.mkdir(parents=True, exist_ok=True)
    title_font = plots.use_style()
    metrics, pred = plots.load_results("chronological")
    random_metrics, _ = plots.load_results("random")
    prepared = pd.read_csv(ROOT / "results" / "chronological" / "prepared_data.csv")

    save_one("classification_roc.png", (8.2, 5.4), plots.draw_roc, pred, metrics)
    save_one("classification_confusion_matrix.png", (5.6, 5.0), plots.draw_confusion, pred, metrics)
    save_one("regression_rmse.png", (8.2, 4.6), plots.draw_rmse, metrics)
    save_one("regression_observed_predicted.png", (7.2, 6.0), plots.draw_observed_predicted, pred, metrics)
    save_one("protocol_comparison.png", (7.6, 4.8), plots.draw_protocols, metrics, random_metrics)
    save_one("timeline_splits.png", (11.5, 4.4), plots.draw_timeline, prepared)

    pdf_path = OUT / "kilauea_panel_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
        fig.suptitle("Kilauea earthquake catalog: chronological test results", fontsize=18,
                     fontweight="bold", color=plots.INK, fontfamily=title_font)
        plots.draw_roc(axes[0, 0], pred, metrics)
        plots.draw_confusion(axes[0, 1], pred, metrics)
        plots.draw_rmse(axes[1, 0], metrics)
        plots.draw_observed_predicted(axes[1, 1], pred, metrics)
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(OUT / "kilauea_panel_figures.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"Figures saved in {OUT}")


if __name__ == "__main__":
    create()

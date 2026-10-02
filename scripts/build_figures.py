"""Create panel-ready charts from the saved chronological test results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "chronological"
OUT = ROOT / "output" / "figures"
COLORS = {"navy":"#123047", "green":"#087E8B", "orange":"#D97732", "blue":"#4C78A8",
          "red":"#C44E52", "gray":"#7B8790", "grid":"#DCE5EA"}
FAMILIES = ["baseline", "logistic", "forest", "kmeans", "mlp"]
LABELS = {"baseline":"Prior baseline", "logistic":"Logistic", "forest":"Random forest",
          "kmeans":"K-means", "mlp":"Neural network"}


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["grid"])
    ax.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["navy"])


def draw_roc(ax, pred, y):
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]]
    for family, color in zip(FAMILIES, colors):
        score = pred[f"classification_{family}"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{LABELS[family]} ({auc:.3f})")
    ax.plot([0,1],[0,1], color=COLORS["navy"], linestyle="--", linewidth=1.2,
            label="Chance (0.500)")
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="False positive rate", ylabel="True positive rate",
           title="Chronological test: eruption classification ROC")
    ax.text(0.03,0.96,f"Chronological test | n={len(y)}, eruptive={int(y.sum())}",
            transform=ax.transAxes,va="top",color=COLORS["navy"],fontsize=8.5)
    ax.legend(loc="lower right",frameon=False,fontsize=7.5)
    axis_style(ax)


def draw_confusion(ax, pred, y, metrics):
    family = metrics["selection"]["classification"]["family"]
    config = metrics["selection"]["classification"]["config"]
    predicted = (pred[f"classification_{family}"].astype(float).to_numpy() >= .5).astype(int)
    cm = confusion_matrix(y,predicted,labels=[0,1])
    im = ax.imshow(cm,cmap="Blues")
    ax.set_xticks([0,1],labels=["Repose","Erupting"])
    ax.set_yticks([0,1],labels=["Repose","Erupting"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Catalog label")
    ax.set_title(f"Chronological test: confusion matrix ({family}, {config})")
    threshold = cm.max()/2
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",fontsize=17,fontweight="bold",
                    color="white" if cm[i,j] > threshold else COLORS["navy"])
    return im


def draw_rmse(ax, metrics):
    reg=metrics["metrics"]["regression"]
    order=["baseline","forest","kmeans","mlp"]
    selected=metrics["selection"]["regression"]["family"]
    values=[reg[f]["test"]["rmse_days"] for f in order]
    y=np.arange(len(order))
    barcolors=[COLORS["green"] if f==selected else COLORS["blue"] for f in order]
    bars=ax.barh(y,values,color=barcolors,height=.62)
    ax.set_yticks(y,labels=[LABELS[f]+(" *" if f==selected else "") for f in order])
    ax.invert_yaxis()
    ax.set_xlim(0,max(values)*1.22)
    ax.set_xlabel("Test RMSE (days; lower is better)")
    ax.set_title("Chronological test: time-to-eruption error")
    for bar,value in zip(bars,values):
        ax.text(value+max(values)*.02,bar.get_y()+bar.get_height()/2,f"{value:.2f} d",
                va="center",color=COLORS["navy"],fontweight="bold")
    ax.text(.99,.02,"* Selected on development data",transform=ax.transAxes,ha="right",
            va="bottom",color=COLORS["gray"],fontsize=8)
    axis_style(ax)


def draw_tte(ax, pred, metrics):
    observed=pred["tte_days"].astype(float).to_numpy()
    reg=metrics["metrics"]["regression"]
    baseline=np.maximum(0,pred["regression_baseline"].astype(float).to_numpy())
    forest=np.maximum(0,pred["regression_forest"].astype(float).to_numpy())
    upper=max(float(observed.max()),float(baseline.max()),float(forest.max()))*1.04
    ax.scatter(observed,baseline,s=24,alpha=.45,color=COLORS["green"],label="Mean baseline")
    ax.scatter(observed,forest,s=25,alpha=.55,color=COLORS["orange"],label="Forest candidate")
    ax.plot([0,upper],[0,upper],color=COLORS["navy"],linestyle="--",linewidth=1.2,
            label="Perfect prediction")
    ax.set(xlim=(0,upper),ylim=(0,upper),xlabel="Observed (days)",
           ylabel="Predicted (days)",title="Chronological test: observed versus predicted")
    ax.text(.03,.96,"Mean baseline won development selection",transform=ax.transAxes,
            va="top",color=COLORS["navy"],fontsize=8.5)
    ax.legend(loc="lower right",frameon=False,fontsize=8)
    axis_style(ax)


def save_one(filename, draw, *args):
    fig,ax=plt.subplots(figsize=(8.2,5.2),layout="constrained")
    draw(ax,*args)
    fig.savefig(OUT/filename,dpi=300,bbox_inches="tight")
    plt.close(fig)


def create():
    OUT.mkdir(parents=True,exist_ok=True)
    pred=pd.read_csv(RESULTS/"test_predictions.csv")
    metrics=json.loads((RESULTS/"metrics.json").read_text(encoding="utf-8"))
    y=pred["erupting"].astype(int).to_numpy()
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\calibri.ttf")
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\calibrib.ttf")
    plt.rcParams.update({"font.family":"Calibri","font.size":9.5,"axes.titlesize":12,
        "axes.labelsize":9.5,"legend.fontsize":8,"figure.facecolor":"white",
        "axes.facecolor":"white","savefig.facecolor":"white","pdf.fonttype":42,"ps.fonttype":42})

    save_one("classification_roc.png",draw_roc,pred,y)
    fig,ax=plt.subplots(figsize=(6.4,5.2),layout="constrained")
    im=draw_confusion(ax,pred,y,metrics)
    fig.colorbar(im,ax=ax,shrink=.82,label="Earthquake count")
    fig.savefig(OUT/"classification_confusion_matrix.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    save_one("regression_rmse.png",draw_rmse,metrics)
    save_one("regression_observed_predicted.png",draw_tte,pred,metrics)

    pdf_path=OUT/"kilauea_panel_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(13.33,9.2),layout="constrained")
        fig.suptitle("Kilauea earthquake catalog | chronological test results",
                     fontsize=18,fontweight="bold",color=COLORS["navy"])
        draw_roc(axes[0,0],pred,y)
        im=draw_confusion(axes[0,1],pred,y,metrics)
        fig.colorbar(im,ax=axes[0,1],shrink=.75,label="Count")
        draw_rmse(axes[1,0],metrics)
        draw_tte(axes[1,1],pred,metrics)
        pdf.savefig(fig,bbox_inches="tight")
        fig.savefig(OUT/"kilauea_panel_figures.png",dpi=220,bbox_inches="tight")
        plt.close(fig)
    print(f"Figures saved in {OUT}")
    print(f"Combined panel sheet: {pdf_path}")


if __name__=="__main__":
    create()

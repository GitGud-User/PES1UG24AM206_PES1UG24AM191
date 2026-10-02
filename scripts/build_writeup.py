"""Build the assignment's two-page PDF summary from recorded experiment outputs."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "PEEVED_Mini_Project_Summary.pdf"
RESULT = ROOT / "results" / "chronological" / "metrics.json"
NAVY = colors.HexColor("#174A35")
HEADER_ORANGE = colors.HexColor("#C45A1C")
TEAL = colors.HexColor("#087E8B")
GOLD = colors.HexColor("#E9A23B")
INK = colors.HexColor("#253746")
MUTED = colors.HexColor("#607486")
PALE = colors.HexColor("#EAF1F5")
LIGHT_GREEN = colors.HexColor("#E8F2E8")

pdfmetrics.registerFont(TTFont("Calibri", r"C:\Windows\Fonts\calibri.ttf"))
pdfmetrics.registerFont(TTFont("Calibri-Bold", r"C:\Windows\Fonts\calibrib.ttf"))

styles = {
    "body": ParagraphStyle("body", fontName="Calibri", fontSize=9.4, leading=13.2,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="Calibri", fontSize=8, leading=10.5,
                            textColor=MUTED),
}


def paragraph(c, text, x, top, width, style="body"):
    p = Paragraph(text, styles[style])
    _, height = p.wrap(width, 500)
    p.drawOn(c, x, top - height)
    return top - height


def section(c, title, x, y):
    c.setFillColor(colors.black)
    c.setFont("Calibri-Bold", 12)
    c.drawString(x, y, title)
    return y - 8


def footer(c, n):
    # The user requested a clean page bottom without course or page-number footers.
    pass


def metric(c, x, y, value, label, width=1.55 * inch):
    c.setFillColor(NAVY)
    c.setFont("Calibri-Bold", 18)
    c.drawString(x, y, value)
    c.setFillColor(MUTED)
    c.setFont("Calibri", 8)
    c.drawString(x, y - 13, label)


def build():
    if not RESULT.exists():
        raise SystemExit("Run python -m src.experiment --protocol chronological first")
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    audit = report["audit"]
    cls = report["metrics"]["classification"]["kmeans"]["test"]
    reg = report["metrics"]["regression"]["baseline"]["test"]
    random_report = json.loads((ROOT / "results/random/metrics.json").read_text(encoding="utf-8"))
    random_cls = random_report["metrics"]["classification"]["forest"]["test"]
    random_reg = random_report["metrics"]["regression"]["forest"]["test"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=letter)
    c.setTitle("PEEVED mini-project summary")
    c.setAuthor("Machine Learning Mini-Project Team")

    # Page 1: question, data and method.
    c.setFillColor(HEADER_ORANGE)
    c.rect(0,  letter[1] - 1.45 * inch, letter[0], 1.45 * inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Calibri-Bold", 21)
    c.drawString(0.65 * inch, letter[1] - 0.70 * inch, "Earthquake signals and Kilauea eruptions")
    c.setFont("Calibri", 11)
    c.drawString(0.66 * inch, letter[1] - 1.02 * inch,
                 "A chronological machine-learning study of Pu'u O'o activity")
    c.setFillColor(GOLD)
    c.rect(0, letter[1] - 1.49 * inch, letter[0], 0.04 * inch, fill=1, stroke=0)

    x, width = 0.68 * inch, 7.1 * inch
    y = letter[1] - 1.82 * inch
    y = section(c, "Problem statement", x, y)
    y = paragraph(c,
        "Can the location, depth, magnitude, and recent rate of earthquakes in Kilauea's East Rift Zone "
        "help identify an ongoing Pu'u O'o eruptive episode or estimate the time to the next eruption? "
        "We reproduce these two tasks from the Stanford CS229 study by Mullet, Singh, and Wong, then "
        "evaluate later earthquakes with a time-ordered holdout.", x, y, width) - 13

    y = section(c, "Dataset and labels", x, y)
    y = paragraph(c,
        f"The bundled catalogs contain {audit['raw_earthquakes']:,} earthquake records and "
        f"{audit['raw_eruptions']} eruption entries. Filtering to the East Rift Zone leaves "
        f"{audit['east_rift_earthquakes']:,} earthquakes; after a 30-day feature warm-up, the "
        f"episode-48 coverage cutoff, and removal of records without a known next onset, "
        f"{audit['model_rows']:,} examples remain ({audit['positive_rows']} eruptive, "
        f"{audit['positive_fraction']:.1%}). Features are longitude, latitude, depth, magnitude, "
        "and counts in the preceding 1, 7, and 30 days. Erupting labels use each catalog duration "
        "plus a 24-hour uncertainty buffer; time-to-eruption is measured in days.", x, y, width) - 13

    y = section(c, "Approach and implementation", x, y)
    y = paragraph(c,
        "We compare a prior/mean baseline, logistic regression, random forests, class-conditional "
        "K-means, and a small multilayer perceptron. Scaling is fit on training data. Hyperparameters "
        "are selected on development data, then evaluated once on the held-out test period. The "
        "chronological split uses 70% train, 20% development, and 10% test by time, a 30-day gap, "
        "and label-availability constraints. A random split is retained as a comparison. Code in "
        "src/data.py, src/models.py, and src/experiment.py writes the audit, metrics, predictions, "
        "and model bundles; demo.py predicts on a held-out earthquake.",
        x, y, width)
    y -= 9
    c.setFillColor(LIGHT_GREEN)
    c.roundRect(x, y - 42, width, 42, 5, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont("Calibri-Bold", 9)
    c.drawString(x + 10, y - 16, "Data quality decision")
    paragraph(c,
        "Episode 48 lists an implausible duration of 173,566,800 hours. We retain its onset as an "
        "exclusive cutoff and do not use the corrupted duration. The source also lists eruptions "
        "35.5 and 35 out of order; preprocessing sorts onset-duration pairs together.",
        x + 10, y - 21, width - 20, "small")
    footer(c, 1)
    c.showPage()

    # Page 2: results, interpretation, conclusion and sources.
    c.setFillColor(colors.black)
    c.setFont("Calibri-Bold", 20)
    c.drawString(0.65 * inch, letter[1] - 0.67 * inch, "Results and conclusion")
    c.setFillColor(MUTED)
    c.setFont("Calibri", 10)
    c.drawString(0.66 * inch, letter[1] - 0.94 * inch,
                 "The chronological test period is the primary measure of future-period performance.")

    top = letter[1] - 1.48 * inch
    metric(c, 0.72 * inch, top, f"{cls['auroc']:.3f}", "classification AUROC")
    metric(c, 2.62 * inch, top, f"{cls['kappa']:.3f}", "Cohen's kappa")
    metric(c, 4.52 * inch, top, f"{reg['rmse_days']:.2f} d", "regression RMSE")
    metric(c, 6.35 * inch, top, f"{reg['r2']:.3f}", "regression R-squared")

    x, width = 0.68 * inch, 7.1 * inch
    y = top - 48
    y = section(c, "Held-out classification", x, y)
    y = paragraph(c,
        f"Development selected class-conditional K-means (k=4). On the later test window, AUROC "
        f"was {cls['auroc']:.3f}, average precision {cls['average_precision']:.3f}, and kappa "
        f"{cls['kappa']:.3f}. It identified {cls['confusion_matrix'][1][1]} of "
        f"{sum(cls['confusion_matrix'][1])} eruptive examples (recall {cls['recall']:.3f}) "
        f"while marking {cls['confusion_matrix'][0][1]} repose examples as eruptive (precision "
        f"{cls['precision']:.3f}). The score is near chance and produces many false alarms.",
        x, y, width) - 11

    y = section(c, "Held-out time-to-eruption regression", x, y)
    y = paragraph(c,
        f"Development selected the mean baseline. It achieved RMSE {reg['rmse_days']:.2f} days, "
        f"MAE {reg['mae_days']:.2f} days, and R-squared {reg['r2']:.3f}. All tested regression "
        "families had negative test R-squared, so none explained more variation than a test-period "
        "mean reference. These data do not support useful time-to-eruption forecasts.", x, y, width) - 11

    y = section(c, "What the random split changes", x, y)
    y = paragraph(c,
        f"A random test split gave the forest classifier AUROC {random_cls['auroc']:.3f} and kappa "
        f"{random_cls['kappa']:.3f}; its forest regression had RMSE {random_reg['rmse_days']:.2f} "
        f"days and R-squared {random_reg['r2']:.3f}. Neighboring earthquakes can appear in both "
        "training and test sets under this split. The gap between protocols shows why the "
        "chronological result is the more relevant estimate for later-period use.", x, y, width) - 11

    y = section(c, "Conclusion", x, y)
    y = paragraph(c,
        "With seven catalog features and a small, imbalanced time series, these models do not "
        "forecast eruption timing reliably. The classification score has limited ranking power "
        "and low precision, while regression does not beat a mean reference by R-squared. Future "
        "work should test longer catalogs and independent signals such as ground deformation, gas "
        "emissions, and focal mechanisms. This is a classroom analysis, not an operational hazard tool.",
        x, y, width) - 8

    c.setStrokeColor(PALE)
    c.line(x, y, x + width, y)
    y -= 14
    y = paragraph(c,
        "Sources: Mullet, Singh, and Wong, “Predicting Eruptive Events at Volcanoes from Earthquake "
        "Data,” Stanford CS229, Fall 2019 (supplied PDF); source code and catalogs: "
        "github.com/bmullet/PEEVED. Results reproduced from the bundled CSV files with this project.",
        x, y, width, "small")
    footer(c, 2)
    c.save()
    print(OUT)


if __name__ == "__main__":
    build()

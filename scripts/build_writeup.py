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
NAVY = colors.HexColor("#1D4ED8")
HEADER_ORANGE = colors.HexColor("#0F172A")
TEAL = colors.HexColor("#1D4ED8")
GOLD = colors.HexColor("#EA580C")
INK = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
PALE = colors.HexColor("#E2E8F0")
LIGHT_GREEN = colors.HexColor("#F1F5F9")

TEAM = "Prithvish Adhikari (PES1UG24AM206)  ·  Piyush Shiv (PES1UG24AM191)"
FONT_DIR = ROOT / "assets" / "fonts"
LABELS = {"baseline": "the prior/mean baseline", "logistic": "logistic regression",
          "forest": "a random forest", "kmeans": "class-conditional K-means",
          "mlp": "a small neural network (MLP)", "extra_trees": "Extra Trees",
          "gradient_boosting": "gradient boosting"}
REG_LABELS = {**LABELS, "baseline": "the mean baseline", "kmeans": "joint K-means"}


def register_fonts():
    """Bundled Open Sans/Montserrat (OFL); fall back to built-in Helvetica anywhere."""
    try:
        pdfmetrics.registerFont(TTFont("Body", str(FONT_DIR / "OpenSans-Regular.ttf")))
        pdfmetrics.registerFont(TTFont("Body-Bold", str(FONT_DIR / "OpenSans-SemiBold.ttf")))
        pdfmetrics.registerFont(TTFont("Head", str(FONT_DIR / "Montserrat-Bold.ttf")))
        return "Body", "Body-Bold", "Head"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Helvetica-Bold"


BODY, BOLD, HEAD = register_fonts()

styles = {
    "body": ParagraphStyle("body", fontName=BODY, fontSize=9.4, leading=13.2,
                           textColor=INK, alignment=TA_LEFT, spaceAfter=5),
    "small": ParagraphStyle("small", fontName=BODY, fontSize=8, leading=10.5,
                            textColor=MUTED),
}


def paragraph(c, text, x, top, width, style="body"):
    p = Paragraph(text, styles[style])
    _, height = p.wrap(width, 500)
    p.drawOn(c, x, top - height)
    return top - height


def section(c, title, x, y):
    c.setFillColor(colors.black)
    c.setFont(BOLD, 12)
    c.drawString(x, y, title)
    return y - 6


def footer(c, n):
    # The user requested a clean page bottom without course or page-number footers.
    pass


def metric(c, x, y, value, label, width=1.55 * inch):
    c.setFillColor(NAVY)
    c.setFont(BOLD, 18)
    c.drawString(x, y, value)
    c.setFillColor(MUTED)
    c.setFont(BODY, 8)
    c.drawString(x, y - 13, label)


def image(c, path, x, top, width, caption):
    """Draw an image at full width with preserved aspect ratio; return the new y."""
    from reportlab.lib.utils import ImageReader
    img = ImageReader(str(path))
    w, h = img.getSize()
    height = width * h / w
    c.drawImage(img, x, top - height, width, height, mask="auto")
    y = top - height - 4
    if caption:
        y = paragraph(c, caption, x, y, width, "small")
    return y


def build():
    if not RESULT.exists():
        raise SystemExit("Run python -m src.experiment --protocol chronological first")
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    audit = report["audit"]
    sel_c = report["selection"]["classification"]
    base_c = report["selection_baseline"]["classification"]
    recall10 = report["selection_recall_first"]["recall_first_fpr_0_10"]
    recall20 = report["selection_recall_first"]["recall_first_fpr_0_20"]
    km = report["selection_kmeans_controlled"]
    km_improvement = report["selection_kmeans_improvement"]
    sel_r = report["selection"]["regression"]
    cls = report["metrics"]["classification"][sel_c["family"]]["test"]
    reg = report["metrics"]["regression"][sel_r["family"]]["test"]
    reg_all = report["metrics"]["regression"]
    all_negative = all(v["test"]["r2"] < 0 for v in reg_all.values())
    random_report = json.loads((ROOT / "results/random/metrics.json").read_text(encoding="utf-8"))
    rsel_c = random_report["selection"]["classification"]["family"]
    rsel_r = random_report["selection"]["regression"]["family"]
    random_cls = random_report["metrics"]["classification"][rsel_c]["test"]
    random_reg = random_report["metrics"]["regression"][rsel_r]["test"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=letter)
    c.setTitle("PEEVED mini-project summary")
    c.setAuthor("Prithvish Adhikari, Piyush Shiv")

    # Page 1: question, data and method.
    c.setFillColor(HEADER_ORANGE)
    c.rect(0,  letter[1] - 1.45 * inch, letter[0], 1.45 * inch, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont(HEAD, 20)
    c.drawString(0.65 * inch, letter[1] - 0.70 * inch, "Earthquake signals and Kilauea eruptions")
    c.setFont(BODY, 11)
    c.drawString(0.66 * inch, letter[1] - 0.98 * inch, "UE24CS352A Machine Learning  |  Mini-project write-up")
    c.drawString(0.66 * inch, letter[1] - 1.20 * inch, TEAM)
    c.setFillColor(GOLD)
    c.rect(0, letter[1] - 1.49 * inch, letter[0], 0.04 * inch, fill=1, stroke=0)

    x, width = 0.68 * inch, 7.1 * inch
    y = letter[1] - 1.82 * inch
    y = section(c, "Problem statement", x, y)
    y = paragraph(c,
        "Can the location, depth, magnitude, and recent rate of earthquakes in Kilauea's East Rift Zone "
        "help identify an ongoing Pu'u O'o eruptive episode or estimate the time to the next eruption? "
        "We adapt these two tasks from the Stanford CS229 study by Mullet, Singh, and Wong (PEEVED) "
        "and add a time-ordered holdout that scores later earthquakes only.", x, y, width) - 13

    y = section(c, "Dataset and labels", x, y)
    y = paragraph(c,
        f"The bundled catalogs contain {audit['raw_earthquakes']:,} earthquake records and "
        f"{audit['raw_eruptions']} eruption entries. Filtering to the East Rift Zone leaves "
        f"{audit['east_rift_earthquakes']:,} earthquakes; after a 30-day feature warm-up, the "
        f"episode-48 coverage cutoff, and removal of records without a known next onset, "
        f"{audit['model_rows']:,} examples remain ({audit['positive_rows']} eruptive, "
        f"{audit['positive_fraction']:.1%}). Features are longitude, latitude, depth, magnitude, "
        "and counts in the preceding 1, 7, and 30 days. Erupting labels use each catalog duration "
        "plus a 24-hour uncertainty buffer; time-to-eruption is measured in days. On the modeled rows these "
        "labels match the original PEEVED rule exactly. Applied to all 3,764 East Rift earthquakes that rule "
        "marks 22.4% as erupting (the paper reports about 20%); the lower 10.3% here comes from excluding "
        "282 post-July-1986 events labelled only through episode 48's corrupted duration and the 522 "
        "warm-up earthquakes of January 1983.", x, y, width) - 13

    y = section(c, "Approach and implementation", x, y)
    y = paragraph(c,
        "We compare prior/mean baselines, logistic regression, random forests, Extra Trees, gradient "
        "boosting, class-conditional K-means, and a small multilayer perceptron. The original seven "
        "features remain available; classification also tests four motivated, strictly past-only "
        "temporal features. Scaling is fit on training data. Hyperparameters, feature set, and "
        "classification threshold are selected on development data, then evaluated once on the held-out test period. The "
        "chronological split uses 70% train, 20% development, and 10% test by time, a 30-day gap, "
        "and label-availability constraints. A random split is retained as a comparison. Code in "
        "src/data.py, src/models.py, and src/experiment.py writes the audit, metrics, predictions, "
        "and model bundles; demo.py predicts on a held-out earthquake.",
        x, y, width)
    y -= 9
    c.setFillColor(LIGHT_GREEN)
    c.roundRect(x, y - 42, width, 42, 5, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.setFont(BOLD, 9)
    c.drawString(x + 10, y - 16, "Data quality decision")
    paragraph(c,
        "Episode 48 lists an implausible duration of 173,566,800 hours. We retain its onset as an "
        "exclusive cutoff and do not use the corrupted duration. The source also lists eruptions "
        "35.5 and 35 out of order; preprocessing sorts onset-duration pairs together.",
        x + 10, y - 21, width - 20, "small")
    image(c, ROOT / "output/figures/timeline_splits.png", x, y - 58, width,
          "Figure 1. Weekly East Rift Zone earthquakes with eruptive-episode labels (orange ticks) and the "
          "time-ordered train / development / test windows.")
    footer(c, 1)
    c.showPage()

    # Page 2: results, interpretation, conclusion and sources.
    c.setFillColor(colors.black)
    c.setFont(BOLD, 20)
    c.drawString(0.65 * inch, letter[1] - 0.67 * inch, "Results and conclusion")
    c.setFillColor(MUTED)
    c.setFont(BODY, 10)
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
        f"Development selected {LABELS[sel_c['family']]} with the {sel_c['feature_set'].replace('_', ' ')} "
        f"feature set ({sel_c['config']}) and a development-F1 threshold of {sel_c['threshold']:.3f}. "
        f"On the later test window, AUROC "
        f"was {cls['auroc']:.3f}, average precision {cls['average_precision']:.3f}, and kappa "
        f"{cls['kappa']:.3f}. It identified {cls['confusion_matrix'][1][1]} of "
        f"{sum(cls['confusion_matrix'][1])} eruptive examples (recall {cls['recall']:.3f}) "
        f"while marking {cls['confusion_matrix'][0][1]} repose examples as eruptive (precision "
        f"{cls['precision']:.3f}). At the fixed 0.5 threshold it would identify "
        f"{cls['threshold_0_5']['confusion_matrix'][1][1]} eruptive examples with F1 "
        f"{cls['threshold_0_5']['f1']:.3f}. The ranking score is near chance and the "
        "development-selected operating point produces many false alarms. The retained AUROC baseline "
        f"would instead select {LABELS[base_c['family']]} ({base_c['config']}, "
        f"development AUROC {base_c['development_metrics']['auroc']:.3f}); the two procedures optimize "
        "different objectives, so the F1-selected procedure is not automatically an improvement.",
        x, y, width) - 11

    y = section(c, "Recall-first constraints", x, y)
    y = paragraph(c,
        f"Under a development FPR cap of 0.10, recall-first selected "
        f"{LABELS[recall10['family']]} at threshold {recall10['threshold']:.3f}; its development "
        f"recall/FPR were {recall10['development_metrics']['recall']:.3f}/{recall10['development_metrics']['false_positive_rate']:.3f}, "
        f"and its test recall/FPR were {recall10['test_metrics']['recall']:.3f}/{recall10['test_metrics']['false_positive_rate']:.3f}. "
        f"Under a 0.20 cap it selected {LABELS[recall20['family']]} at threshold {recall20['threshold']:.3f}; "
        f"test recall/FPR were {recall20['test_metrics']['recall']:.3f}/{recall20['test_metrics']['false_positive_rate']:.3f}. "
        "The test FPR is not constrained because it was not used for selection, and neither strategy recovered "
        "the historical K-means positive count.",
        x, y, width) - 11

    y = section(c, "Controlled K-means threshold experiment", x, y)
    y = paragraph(c,
        f"To isolate the historical model from model selection, we fixed class-conditional K-means "
        f"(k=4) and the original seven features. The fixed 0.5 operating point found "
        f"{km['original_fixed_0_5']['test_metrics']['true_positives']} of 19 test positives "
        f"(recall {km['original_fixed_0_5']['test_metrics']['recall']:.3f}) with "
        f"{km['original_fixed_0_5']['test_metrics']['false_positives']} false positives. "
        f"Development-F1 selection raised the threshold to {km['development_f1']['threshold']:.3f} "
        f"and found {km['development_f1']['test_metrics']['true_positives']} positives. "
        f"The development-FPR-constrained operating points found "
        f"{km['recall_first_fpr_0_10']['test_metrics']['true_positives']} and "
        f"{km['recall_first_fpr_0_20']['test_metrics']['true_positives']} positives for caps "
        f"0.10 and 0.20, respectively. These thresholds were selected without test labels; "
        f"the K-means test AUROC and average precision are {km['original_fixed_0_5']['test_metrics']['auroc']:.3f} "
        f"and {km['original_fixed_0_5']['test_metrics']['average_precision']:.3f} for every row because "
        "thresholding changes decisions, not ranking scores. The fixed historical threshold and optimized "
        "thresholds are different operating procedures, and the results do not establish an improvement "
        "over the historical result.",
        x, y, width) - 11

    y = section(c, "Controlled K-means model improvement", x, y)
    y = paragraph(c,
        f"We then evaluated k in 2, 3, 4, 5, 6 and 8 with raw and train-standardized "
        f"original-seven-feature inputs and two positively oriented distance scores. "
        f"Development AUROC selected {km_improvement['ranking_model']['config']} "
        f"(development AUROC {km_improvement['ranking_model']['development_metrics']['auroc']:.3f}, "
        f"average precision {km_improvement['ranking_model']['development_metrics']['average_precision']:.3f}). "
        f"At its development-F1 threshold, the frozen test AUROC/AP were "
        f"{km_improvement['operating_points']['development_f1']['test_metrics']['auroc']:.3f}/"
        f"{km_improvement['operating_points']['development_f1']['test_metrics']['average_precision']:.3f}, "
        f"with recall {km_improvement['operating_points']['development_f1']['test_metrics']['recall']:.3f} "
        f"and FPR {km_improvement['operating_points']['development_f1']['test_metrics']['false_positive_rate']:.3f}. "
        "This is a development-selected ranking configuration, not evidence of reliable future-period "
        "forecasting; the test ranking can differ substantially from development performance.",
        x, y, width) - 11

    y = section(c, "Held-out time-to-eruption regression", x, y)
    y = paragraph(c,
        f"Development selected {REG_LABELS[sel_r['family']]}{'' if sel_r['family'] == 'baseline' else ' (' + sel_r['config'] + ')'}. It achieved RMSE "
        f"{reg['rmse_days']:.2f} days, MAE {reg['mae_days']:.2f} days, and R-squared {reg['r2']:.3f}. "
        + ("All tested regression families had negative test R-squared, so none explained more variation "
           "than a test-period mean reference. " if all_negative else "")
        + "These data do not support useful time-to-eruption forecasts.", x, y, width) - 11

    y = section(c, "What the random split changes", x, y)
    y = paragraph(c,
        f"With a stratified random 70/20/10 split (as in the paper), development selected "
        f"{LABELS[rsel_c]} for classification (test AUROC {random_cls['auroc']:.3f}, kappa "
        f"{random_cls['kappa']:.3f}) and {REG_LABELS[rsel_r]} for regression (RMSE "
        f"{random_reg['rmse_days']:.2f} days, R-squared {random_reg['r2']:.3f}). The paper's random "
        "forest reported test kappa 0.52 and AUROC 0.71 (its text quotes kappa 0.42) and R-squared 0.38 "
        "on a standardized target. Neighboring earthquakes can appear in both training and test sets "
        "under a random split, so it is not an estimate of performance on later periods.", x, y, width) - 11

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
        "Data,” Stanford CS229, Fall 2019; original code and catalogs: github.com/bmullet/PEEVED "
        "(earthquakes: ANSS catalog via WOVOdat; eruptions: Hawaii Center for Volcanology). Results are "
        "this project's own runs on those bundled files (scikit-learn 1.7.2).",
        x, y, width, "small")
    half = (width - 14) / 2
    top = y - 12
    y1 = image(c, ROOT / "output/figures/classification_roc.png", x, top, half, None)
    y2 = image(c, ROOT / "output/figures/protocol_comparison.png", x + half + 14, top, half, None)
    paragraph(c, "Figure 2. Left: chronological-test ROC curves (all models near the chance line). Right: "
              "selected-model test metrics under the chronological and random splits.",
              x, min(y1, y2) - 2, width, "small")
    footer(c, 2)
    c.save()
    print(OUT)


if __name__ == "__main__":
    build()

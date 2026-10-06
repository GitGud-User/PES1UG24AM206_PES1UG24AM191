"""Build graphs.html, the results page, from saved results and figures (no retraining).

    python scripts/build_site.py              # graphs.html next to README (relative links)
    python scripts/build_site.py --embed OUT  # single self-contained file (images inlined)
"""
import argparse
import base64
import html
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import past_counts  # noqa: E402
from src.experiment import split_data  # noqa: E402

CLASS = {"baseline": "Prior baseline", "logistic": "Logistic regression", "forest": "Random forest",
         "kmeans": "Class-conditional K-means", "mlp": "Neural network (MLP)"}
REG = {"baseline": "Mean baseline", "forest": "Random forest", "kmeans": "Joint K-means",
       "mlp": "Neural network (MLP)"}
TEAM = [("Prithvish Adhikari", "PES1UG24AM206"), ("Piyush Shiv", "PES1UG24AM191")]

CSS = """
/* Layout: one reading column (max 72rem) with full-width figure plates; results first, method after. */
:root {
  --bg: #F8FAFC; --surface: #FFFFFF; --ink: #0F172A; --text: #334155; --muted: #64748B;
  --line: #E2E8F0; --primary: #1D4ED8; --accent: #EA580C; --plate: #FFFFFF; --code-bg: #F1F5F9;
  --chip: #EFF6FF; --chip-ink: #1E40AF;
  --display: "Montserrat", "Arial Black", "Segoe UI", sans-serif;
  --body: "Open Sans", "Segoe UI", Calibri, Arial, sans-serif;
  --mono: ui-monospace, "Cascadia Code", Consolas, "SFMono-Regular", monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0B1120; --surface: #111827; --ink: #F1F5F9; --text: #CBD5E1; --muted: #94A3B8;
  --line: #1E293B; --primary: #60A5FA; --accent: #FB923C; --plate: #FFFFFF; --code-bg: #0F172A;
  --chip: #172554; --chip-ink: #BFDBFE; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0B1120; --surface: #111827; --ink: #F1F5F9; --text: #CBD5E1; --muted: #94A3B8;
  --line: #1E293B; --primary: #60A5FA; --accent: #FB923C; --plate: #FFFFFF; --code-bg: #0F172A;
  --chip: #172554; --chip-ink: #BFDBFE; color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.65 var(--body); }
.wrap { max-width: 72rem; margin: 0 auto; padding-inline: clamp(16px, 4vw, 40px); }
header.top { border-bottom: 1px solid var(--line); background: var(--surface); }
header.top .wrap { padding-block: 40px 32px; display: grid; gap: 14px; }
.eyebrow { font: 600 12px/1.2 var(--body); letter-spacing: .12em; text-transform: uppercase; color: var(--accent); }
h1, h2, h3 { font-family: var(--display); color: var(--ink); text-wrap: balance; margin: 0; }
h1 { font-size: clamp(30px, 5vw, 48px); line-height: 1.08; font-weight: 800; letter-spacing: -.01em; }
h2 { font-size: clamp(22px, 3vw, 28px); line-height: 1.2; font-weight: 700; }
h3 { font-size: 17px; font-weight: 700; }
.lede { max-width: 62ch; font-size: 18px; color: var(--text); margin: 0; }
.team { display: flex; flex-wrap: wrap; gap: 8px 18px; font-size: 14px; color: var(--muted); }
.team b { color: var(--ink); font-weight: 600; }
nav.toc { display: flex; flex-wrap: wrap; gap: 8px; padding-block: 18px; }
nav.toc a { font-size: 14px; font-weight: 600; color: var(--primary); text-decoration: none; padding: 6px 12px;
  border: 1px solid var(--line); border-radius: 999px; background: var(--surface); }
nav.toc a:hover { border-color: var(--primary); }
a:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
main .wrap { display: grid; gap: 56px; padding-block: 24px 64px; }
section { display: grid; gap: 18px; min-width: 0; }
section > *, .two > *, main .wrap > * { min-width: 0; }
section > p, .note { max-width: 70ch; margin: 0; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: var(--line);
  border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
.kpi { background: var(--surface); padding: 18px 20px; display: grid; gap: 2px; }
.kpi .v { font: 800 30px/1.1 var(--display); color: var(--ink); font-variant-numeric: tabular-nums; }
.kpi .l { font-size: 13px; color: var(--muted); }
.kpi .c { font-size: 12px; color: var(--muted); }
.callout { border-left: 3px solid var(--accent); padding: 4px 0 4px 16px; max-width: 70ch; color: var(--text); }
figure { margin: 0; display: grid; gap: 8px; min-width: 0; }
figure a { display: block; background: var(--plate); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
figure img { display: block; width: 100%; height: auto; max-width: 100%; }
figcaption { font-size: 13.5px; color: var(--muted); max-width: 75ch; }
.two { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 24px; align-items: start; }
.two.even { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.stack { display: grid; gap: 20px; }
.stack figure { max-width: 46rem; }
.tbl { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 14px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { font-weight: 600; color: var(--muted); font-size: 12.5px; letter-spacing: .03em; }
tbody tr:last-child td { border-bottom: 0; }
tr.sel td { color: var(--ink); font-weight: 600; }
.pill { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
  background: var(--chip); color: var(--chip-ink); border-radius: 999px; padding: 2px 8px; margin-left: 6px; }
ol.steps { margin: 0; padding: 0; list-style: none; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; counter-reset: s; }
ol.steps li { counter-increment: s; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; font-size: 14.5px; min-width: 0; }
ol.steps li::before { content: "Step " counter(s); display: block; font: 700 12px var(--body); color: var(--accent); letter-spacing: .06em; text-transform: uppercase; margin-bottom: 4px; }
ol.steps b { color: var(--ink); }
pre { margin: 0; background: var(--code-bg); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;
  overflow-x: auto; font: 13px/1.55 var(--mono); color: var(--ink); }
code { font-family: var(--mono); font-size: .92em; }
details { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
details summary { cursor: pointer; padding: 12px 16px; font-weight: 600; color: var(--ink); }
details pre { border: 0; border-top: 1px solid var(--line); border-radius: 0 0 10px 10px; }
footer { border-top: 1px solid var(--line); }
footer .wrap { padding-block: 24px 40px; font-size: 13.5px; color: var(--muted); display: grid; gap: 6px; }
@media (max-width: 860px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } .two, .two.even { grid-template-columns: 1fr; }
  ol.steps { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 480px) { ol.steps { grid-template-columns: 1fr; } .kpi .v { font-size: 26px; } }
"""


def fmt(v, d=3):
    return f"{v:.{d}f}".replace("-", "−")


def img_src(name, embed):
    path = ROOT / "output" / "figures" / name
    if embed:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()
    return f"output/figures/{name}"


def figure(name, alt, caption, embed):
    src = img_src(name, embed)
    link = "" if embed else f' href="{src}" target="_blank" rel="noopener"'
    tag = "a" if not embed else "div"
    style = "" if not embed else ' style="background:var(--plate);border:1px solid var(--line);border-radius:10px;padding:10px"'
    return (f'<figure><{tag}{link}{style}><img src="{src}" alt="{html.escape(alt)}"></{tag}>'
            f"<figcaption>{caption}</figcaption></figure>")


def build(embed=False):
    chrono = json.loads((ROOT / "results/chronological/metrics.json").read_text(encoding="utf-8"))
    rnd = json.loads((ROOT / "results/random/metrics.json").read_text(encoding="utf-8"))
    a = chrono["audit"]
    sc, sr = chrono["selection"]["classification"], chrono["selection"]["regression"]
    ct = chrono["metrics"]["classification"][sc["family"]]["test"]
    rt = chrono["metrics"]["regression"][sr["family"]]["test"]
    cm = ct["confusion_matrix"]
    rsc, rsr = rnd["selection"]["classification"]["family"], rnd["selection"]["regression"]["family"]
    rct = rnd["metrics"]["classification"][rsc]["test"]
    rrt = rnd["metrics"]["regression"][rsr]["test"]
    sp = a["splits"]

    cls_rows = "".join(
        f'<tr class="{"sel" if f == sc["family"] else ""}"><td>{CLASS[f]}'
        f'{"<span class=pill>selected</span>" if f == sc["family"] else ""}</td><td>{v["config"]}</td>'
        f'<td>{fmt(v["development"]["auroc"])}</td><td>{fmt(v["test"]["auroc"])}</td>'
        f'<td>{fmt(v["test"]["average_precision"])}</td><td>{fmt(v["test"]["kappa"])}</td>'
        f'<td>{fmt(v["test"]["recall"])}</td><td>{fmt(v["test"]["precision"])}</td></tr>'
        for f, v in chrono["metrics"]["classification"].items())
    reg_rows = "".join(
        f'<tr class="{"sel" if f == sr["family"] else ""}"><td>{REG[f]}'
        f'{"<span class=pill>selected</span>" if f == sr["family"] else ""}</td><td>{v["config"]}</td>'
        f'<td>{fmt(v["development"]["rmse_days"], 2)}</td><td>{fmt(v["test"]["rmse_days"], 2)}</td>'
        f'<td>{fmt(v["test"]["mae_days"], 2)}</td><td>{fmt(v["test"]["r2"])}</td></tr>'
        for f, v in chrono["metrics"]["regression"].items())

    split_src = inspect.getsource(split_data)
    split_src = split_src[split_src.index("    else:"):split_src.index("    if any(")].rstrip()
    counts_src = inspect.getsource(past_counts).rstrip()
    select_src = '''for family, config, model in candidates(task, seed):
    model.fit(X.iloc[train], y.iloc[train])          # train window only
    metric = score(model, X.iloc[dev], y.iloc[dev])  # development window
    keep the best config per family
winner = family with the best development score     # frozen before any test metric
test metrics are computed once, for every family's dev-best model'''

    fonts = ("" if not embed else
             '<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700;800&'
             'family=Open+Sans:wght@400;600;700&display=swap">')
    local_fonts = "" if embed else """
@font-face { font-family: "Montserrat"; font-weight: 700 800; src: url("assets/fonts/Montserrat-Bold.ttf") format("truetype"); }
@font-face { font-family: "Open Sans"; font-weight: 400; src: url("assets/fonts/OpenSans-Regular.ttf") format("truetype"); }
@font-face { font-family: "Open Sans"; font-weight: 600; src: url("assets/fonts/OpenSans-SemiBold.ttf") format("truetype"); }
@font-face { font-family: "Open Sans"; font-weight: 700; src: url("assets/fonts/OpenSans-Bold.ttf") format("truetype"); }"""
    team = "".join(f"<span><b>{n}</b> · {s}</span>" for n, s in TEAM)

    body = f"""
<header class="top"><div class="wrap">
  <div class="eyebrow">UE24CS352A Machine Learning · Mini-project</div>
  <h1>Earthquake signals and Kilauea eruptions</h1>
  <p class="lede">Can the location, depth, magnitude and recent rate of East Rift Zone earthquakes tell us whether Pu'u 'Ō'ō is erupting, or how many days remain until the next eruptive episode? We adapt the Stanford CS229 PEEVED study and test it on a later time period.</p>
  <div class="team">{team}</div>
</div></header>
<div class="wrap"><nav class="toc" aria-label="Sections">
  <a href="#results">Results</a><a href="#data">Data</a><a href="#method">Method</a>
  <a href="#classification">Classification</a><a href="#regression">Regression</a>
  <a href="#splits">Random vs time-ordered</a><a href="#code">Code</a><a href="#reproduce">Reproduce</a>
</nav></div>
<main><div class="wrap">

<section id="results" aria-labelledby="h-results">
  <h2 id="h-results">Results on the held-out test period</h2>
  <div class="kpis">
    <div class="kpi"><span class="v">{fmt(ct['auroc'])}</span><span class="l">Classification AUROC</span><span class="c">0.5 = chance</span></div>
    <div class="kpi"><span class="v">{fmt(ct['kappa'])}</span><span class="l">Cohen's kappa</span><span class="c">0 = chance agreement</span></div>
    <div class="kpi"><span class="v">{fmt(rt['rmse_days'], 2)} d</span><span class="l">Time-to-eruption RMSE</span><span class="c">{REG[sr['family']].lower()}</span></div>
    <div class="kpi"><span class="v">{fmt(rt['r2'])}</span><span class="l">Regression R²</span><span class="c">below 0 = worse than the test mean</span></div>
  </div>
  <p class="callout">On earthquakes from February to July 1986, the models score close to chance. The development-selected classifier ({CLASS[sc['family']]}, {sc['config']}) found {cm[1][1]} of {sum(cm[1])} eruptive earthquakes but flagged {cm[0][1]} of {sum(cm[0])} repose earthquakes. These are classroom results on historical data, not eruption forecasts or calibrated hazard probabilities.</p>
</section>

<section id="data" aria-labelledby="h-data">
  <h2 id="h-data">Data</h2>
  <p>The bundled catalogs (identical to the original PEEVED files) hold {a['raw_earthquakes']:,} ANSS earthquakes and {a['raw_eruptions']} eruption entries from the Hawaii Center for Volcanology. Filtering to the East Rift Zone leaves {a['east_rift_earthquakes']:,} earthquakes, the same count as the paper. After a 30-day warm-up for the rate features and the end of reliable eruption coverage, <b>{a['model_rows']:,} earthquakes</b> remain, {a['positive_rows']} of them ({a['positive_fraction']:.1%}) during an eruptive episode.</p>
  {figure('timeline_splits.png', 'Bar chart of weekly earthquake counts from 1983 to mid-1986 with shaded train, development and test windows and orange ticks for earthquakes during eruptive episodes.', 'Weekly earthquake counts. Orange ticks mark earthquakes labelled as erupting; shading shows the time-ordered train, development and test windows.', embed)}
  <div class="two even">
    <p class="note"><b>Episode 48 duration.</b> The catalog lists 173,566,800 hours for episode 48 (about 19,800 years). We keep its onset, 20 July 1986, as the end of coverage and never use the duration. The original code used it as-is, which labels every later earthquake as erupting; applying that rule to these files, 282 positive labels come from that row alone.</p>
    <p class="note"><b>Label check.</b> On all {a['model_rows']:,} modeled earthquakes our labels match the original PEEVED rule exactly. That rule labels 22.4% of the {a['east_rift_earthquakes']:,} earthquakes as erupting; excluding the episode-48 rows and the 522 warm-up earthquakes of January 1983 gives our {a['positive_fraction']:.1%}.</p>
  </div>
</section>

<section id="method" aria-labelledby="h-method">
  <h2 id="h-method">Method</h2>
  <ol class="steps">
    <li><b>Features.</b> Longitude, latitude, depth, magnitude, and the number of earthquakes in the previous 1, 7 and 30 days, counted strictly before each event.</li>
    <li><b>Labels.</b> Erupting if inside an episode's duration plus a 24-hour buffer; time to eruption in days to the next onset.</li>
    <li><b>Split by time.</b> Train {sp['train']['rows']:,} · development {sp['dev']['rows']:,} · test {sp['test']['rows']:,} earthquakes, with 30-day gaps and no labels that leak across a boundary.</li>
    <li><b>Select, then test once.</b> Baselines, logistic regression, random forests, K-means and an MLP are tuned on development data; the winner is fixed before the test period is scored.</li>
  </ol>
</section>

<section id="classification" aria-labelledby="h-cls">
  <h2 id="h-cls">Is the volcano erupting?</h2>
  <div class="two">
    {figure('classification_roc.png', 'ROC curves for five classifiers on the chronological test period, all close to the diagonal chance line.', 'ROC curves on the test period. All models stay close to the chance diagonal; the random forest falls below it.', embed)}
    {figure('classification_confusion_matrix.png', f'Confusion matrix: {cm[0][0]} repose correct, {cm[0][1]} repose flagged as erupting, {cm[1][0]} eruptive missed, {cm[1][1]} eruptive found.', 'Confusion matrix of the selected model at a 0.5 score threshold.', embed)}
  </div>
  <div class="tbl"><table>
    <thead><tr><th>Model</th><th>Setting</th><th>Dev AUROC</th><th>Test AUROC</th><th>Avg. precision</th><th>Kappa</th><th>Recall</th><th>Precision</th></tr></thead>
    <tbody>{cls_rows}</tbody></table></div>
</section>

<section id="regression" aria-labelledby="h-reg">
  <h2 id="h-reg">How many days until the next eruption?</h2>
  <div class="two even">
    {figure('regression_rmse.png', 'Horizontal bars of test RMSE in days for four regression models, between 7.3 and 8.3 days.', 'Test error by model. The forest is slightly lower than the selected mean baseline, but development data chose the baseline and we keep that choice.', embed)}
    {figure('regression_observed_predicted.png', 'Scatter of predicted versus observed days to eruption; most predictions sit in a flat band between about 11 and 14 days.', 'Predictions barely vary with the observed time to eruption.', embed)}
  </div>
  <div class="tbl"><table>
    <thead><tr><th>Model</th><th>Setting</th><th>Dev RMSE (d)</th><th>Test RMSE (d)</th><th>Test MAE (d)</th><th>Test R²</th></tr></thead>
    <tbody>{reg_rows}</tbody></table></div>
</section>

<section id="splits" aria-labelledby="h-splits">
  <h2 id="h-splits">Why a random split looks better</h2>
  <p>The paper split earthquakes at random (stratified 70/20/10). Earthquakes from the same swarm then appear in both training and test sets, so the test is easier than predicting a later period. Running our pipeline with that split selects random forests and gives higher scores.</p>
  <div class="stack">
    {figure('protocol_comparison.png', 'Grouped bars comparing AUROC, kappa and R squared for the chronological and random splits.', 'Selected-model test metrics under each split. Each split selects its own models on its own development data.', embed)}
    <div class="tbl"><table>
      <thead><tr><th>Test metric</th><th>Time-ordered</th><th>Random (ours)</th><th>Paper (random)</th></tr></thead>
      <tbody>
        <tr><td>Classifier</td><td>{CLASS[sc['family']]}</td><td>{CLASS[rsc]}</td><td>Random forest</td></tr>
        <tr><td>AUROC</td><td>{fmt(ct['auroc'])}</td><td>{fmt(rct['auroc'])}</td><td>0.71</td></tr>
        <tr><td>Kappa</td><td>{fmt(ct['kappa'])}</td><td>{fmt(rct['kappa'])}</td><td>0.52*</td></tr>
        <tr><td>Regressor</td><td>{REG[sr['family']]}</td><td>{REG[rsr]}</td><td>Random forest</td></tr>
        <tr><td>RMSE</td><td>{fmt(rt['rmse_days'], 2)} d</td><td>{fmt(rrt['rmse_days'], 2)} d</td><td>0.64 (scaled)</td></tr>
        <tr><td>R²</td><td>{fmt(rt['r2'])}</td><td>{fmt(rrt['r2'])}</td><td>0.38</td></tr>
      </tbody></table></div>
  </div>
  <p class="note" style="font-size:13.5px;color:var(--muted)">* Paper figures 4 and 5, test set. The paper's text quotes kappa 0.42. Its RMSE is on a standardized target and cannot be compared with RMSE in days. With only {sp['test']['positive_rows']} eruptive earthquakes in our test period, a bootstrap 95% interval for the AUROC spans roughly 0.37–0.68.</p>
</section>

<section id="code" aria-labelledby="h-code">
  <h2 id="h-code">Key code</h2>
  <details open><summary>Past-only earthquake counts · <code>src/data.py</code></summary><pre><code>{html.escape(counts_src)}</code></pre></details>
  <details><summary>Time-ordered split with gaps · <code>src/experiment.py</code></summary><pre><code>{html.escape(split_src)}</code></pre></details>
  <details><summary>Model selection (outline) · <code>src/experiment.py</code></summary><pre><code>{html.escape(select_src)}</code></pre></details>
</section>

<section id="reproduce" aria-labelledby="h-rep">
  <h2 id="h-rep">Reproduce</h2>
  <p>Python 3.12 with the pinned <code>requirements.txt</code> (scikit-learn 1.7.2). The notebook <code>notebooks/walkthrough.ipynb</code> walks through every step with saved outputs.</p>
<pre><code>python -m unittest discover -s tests -v          # pipeline checks
python -m src.experiment --protocol both         # retrain and evaluate (~30 s)
python demo.py --row 0                           # score one held-out earthquake
python scripts/build_figures.py                  # charts from saved results
python scripts/build_site.py                     # this page</code></pre>
</section>

</div></main>
<footer><div class="wrap">
  <span>Source study: B. Mullet, A. Singh and Y.-Q. Wong, “Predicting Eruptive Events at Volcanoes from Earthquake Data,” Stanford CS229 project report, Fall 2019. Original code and catalogs: <a href="https://github.com/bmullet/PEEVED" rel="noopener">github.com/bmullet/PEEVED</a>.</span>
  <span>All numbers on this page are read from <code>results/</code>; scores are not calibrated hazard probabilities.</span>
</div></footer>
"""
    title = "<title>Kilauea Eruption Signals</title>"
    style = f"<style>{local_fonts}{CSS}</style>"
    if embed:
        return f"{title}\n{fonts}\n{style}\n{body}"
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"{title}\n{style}\n</head>\n<body>\n{body}\n</body>\n</html>\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embed", type=Path, help="write a self-contained page (images inlined) here")
    args = ap.parse_args()
    if args.embed:
        args.embed.write_text(build(embed=True), encoding="utf-8")
        print(args.embed)
    else:
        (ROOT / "graphs.html").write_text(build(), encoding="utf-8")
        print(ROOT / "graphs.html")

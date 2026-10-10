# Problem 64: Predicting Eruptive Events at Volcanoes from Earthquake Data

**UE24CS352A Machine Learning mini-project**  
**Team:** Prithvish Adhikari (PES1UG24AM206) and Piyush Shiv (PES1UG24AM191)

This project is inspired by the Stanford CS229 PEEVED study of Kilauea earthquake data. It asks two related questions:

1. **Classification:** did an earthquake occur during an eruptive episode?
2. **Regression:** how many days remained until the next eruption onset?

This is a classroom research project. It is not an operational volcano-warning system, and its scores are not calibrated hazard probabilities.

## Motivation

Earthquake activity can change during volcanic unrest. Location, depth, magnitude, and recent earthquake rates may therefore contain information about whether an eruptive episode is active. The project tests that intuition while emphasizing a realistic time-ordered evaluation: models are trained on earlier earthquakes, selected on a later development period, and evaluated once on a final later test period.

## Dataset

The repository includes the earthquake and eruption catalogs in [`data/raw/`](data/raw/):

- `puuoo_earthquakes.csv`
- `PuuOo.csv`

`src/data.py` filters earthquake records to Kilauea's East Rift Zone, removes invalid or duplicate records, aligns the eruption catalog, and applies the documented episode-48 coverage cutoff. The modeled dataset has **2,960 earthquakes**, including **304 eruptive examples**.

The original seven features are:

- longitude
- latitude
- depth
- magnitude
- earthquake counts in the preceding 1, 7, and 30 days

The expanded classification feature set adds four strictly past-only features:

- earthquake count in the preceding 6 hours
- hours since the previous earthquake
- mean magnitude in the preceding 24 hours
- maximum magnitude in the preceding 24 hours

For every temporal feature, only earthquakes strictly before the prediction timestamp are used. Duplicate timestamps do not contribute to one another. The eruption labels use the catalog episode intervals plus the project's 24-hour uncertainty buffer. The corrupted duration listed for episode 48 (`173566800 hrs`) is not used; its onset is retained as an exclusive coverage cutoff. Eruption entries 35.5 and 35 are sorted with their matching durations before labeling.

## Methodology

### Preprocessing and labels

The data preparation and feature calculations are implemented in [`src/data.py`](src/data.py). The target columns are kept separate from the feature columns, and the time-to-eruption target is not used as a classification feature.

### Chronological evaluation

The primary protocol in [`src/experiment.py`](src/experiment.py) is time ordered:

- training: 2,023 rows
- development: 519 rows
- test: 296 rows

The split uses approximately 70%/20%/10% of the time range, with 30-day temporal gaps and label-availability checks at the boundaries. The development set is used to choose models, configurations, and classification thresholds. The chronological test set is used only after those choices are frozen.

The random split is retained as a secondary comparison. It is easier to overestimate performance because nearby earthquakes can appear in both training and test sets.

## Models and experiments

Implemented model families include:

- prior and mean baselines
- logistic regression
- random forest
- Extra Trees
- gradient boosting
- class-conditional K-means
- small MLP models
- joint K-means regression

All preprocessing in model pipelines is fitted on training rows and then reused for development, test, and inference.

### Classification selection and threshold experiments

The repository records several distinct procedures:

- **AUROC baseline:** selects the candidate with the best development AUROC.
- **Joint development-F1 selection:** selects model, feature set, and threshold primarily by development F1, then average precision and deterministic tie-breakers.
- **Recall-first selection:** selects the highest development recall under development FPR caps of 0.10 and 0.20. Ties prefer fewer false positives, higher precision, higher average precision, higher threshold, and stable names.
- **Controlled K-means threshold comparison:** fixes the existing `k=4`, original-seven-feature K-means model and compares threshold 0.5, development-F1 thresholding, and the two FPR-constrained thresholds.
- **Controlled K-means configuration search:** evaluates `k` in `{2, 3, 4, 5, 6, 8}`, raw versus standardized original-seven-feature inputs, and two equivalent positively oriented distance scores. Configurations are selected by development AUROC, then development average precision.

These experiments answer different questions. Threshold changes affect precision, recall, and false alarms; they do not change AUROC or average precision for a fixed ranking model. The test set is not used to select any of these settings.

### Regression

Regression candidates are selected using development RMSE. The mean baseline is preserved even when another model has a lower test error, because model selection must be based on development data.

## Results

The values below are read from [`results/chronological/metrics.json`](results/chronological/metrics.json). The chronological test contains only 19 positive examples, so all test classification rates are uncertain.

### Primary chronological classification results

| Procedure | Selected model | Features | Threshold | Dev AUROC | Test AUROC | Test AP | Test precision | Test recall | Test F1 | Test FPR | Test TP/FP/TN/FN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AUROC baseline | Gradient Boosting | Expanded 11 | 0.087 | 0.603 | 0.532 | 0.098 | 0.041 | 0.211 | 0.068 | 0.339 | 4/94/183/15 |
| Joint development F1 | Logistic Regression | Expanded 11 | 0.611 | 0.546 | 0.484 | 0.206 | 0.061 | 0.211 | 0.094 | 0.224 | 4/62/215/15 |
| Recall-first, FPR ≤ 0.10 | MLP | Original 7 | 0.458 | — | 0.513 | 0.098 | 0.038 | 0.053 | 0.044 | 0.090 | 1/25/252/18 |
| Recall-first, FPR ≤ 0.20 | Logistic Regression | Expanded 11 | 0.611 | — | 0.484 | 0.206 | 0.061 | 0.211 | 0.094 | 0.224 | 4/62/215/15 |

The AUROC and joint-F1 rows use different model-selection objectives and should not be described as a single improvement sequence. Neither recall-first strategy recovered the historical K-means count of 12/19 positives.

### Controlled historical K-means threshold comparison

This comparison fixes class-conditional K-means with `k=4`, the original seven features, and the existing standardized pipeline.

| Operating point | Threshold | Dev AUROC/AP | Test AUROC/AP | Test precision | Test recall | Test F1 | Test FPR | Test TP/FP/TN/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical fixed threshold | 0.500 | 0.572/0.144 | 0.524/0.096 | 0.058 | 0.632 | 0.107 | 0.700 | 12/194/83/7 |
| Development-F1 threshold | 0.538 | 0.572/0.144 | 0.524/0.096 | 0.087 | 0.474 | 0.146 | 0.343 | 9/95/182/10 |
| Recall-first, development FPR ≤ 0.10 | 0.608 | 0.572/0.144 | 0.524/0.096 | 0.083 | 0.053 | 0.065 | 0.040 | 1/11/266/18 |
| Recall-first, development FPR ≤ 0.20 | 0.565 | 0.572/0.144 | 0.524/0.096 | 0.136 | 0.316 | 0.190 | 0.137 | 6/38/239/13 |

The historical fixed threshold detects 12/19 test positives, but also flags 194 repose examples. The development-selected thresholds reduce false alarms for this run without improving ranking performance.

### Controlled K-means configuration search

The development-AUROC winner was `k=3, scaling=raw, score=relative_distance`.

| Configuration | Dev AUROC | Dev AP | Test AUROC | Test AP | Test F1 threshold | Test TP/FP/TN/FN | Test recall | Test FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical `k=4`, standard, relative distance | 0.572 | 0.144 | 0.524 | 0.096 | 0.500 | 12/194/83/7 | 0.632 | 0.700 |
| Development-selected `k=3`, raw, relative distance | 0.668 | 0.221 | 0.515 | 0.068 | 0.797 | 3/49/228/16 | 0.158 | 0.177 |

The selected configuration improves development AUROC but not chronological test AUROC or recall. The normalized-margin score is an algebraically equivalent form of the relative-distance score and is retained as a score-direction check, not as an independent source of information.

### Regression and random comparison

The chronological development-selected mean regression baseline has test RMSE **7.75 days**, MAE **6.60 days**, and R² **-0.179**. The random-split joint-F1 classification result selects an expanded-feature random forest and reaches test AUROC **0.685**, AP **0.311**, F1 **0.216**, and kappa **0.185**. These random results are secondary and should not be presented as later-period forecasting performance.

## Repository structure

```text
.
├── assets/fonts/                 Fonts used by charts and PDF
├── data/raw/                     Bundled earthquake and eruption catalogs
├── models/                       Saved chronological model bundles
├── notebooks/walkthrough.ipynb   Guided notebook
├── output/
│   ├── figures/                  PNG/PDF figures
│   ├── pdf/                      Two-page report
│   └── slides/                   Final PowerPoint
├── reference/PEEVED/             Original project as a Git submodule
├── results/
│   ├── chronological/            Time-ordered metrics, predictions, tuning
│   └── random/                   Secondary random-split outputs
├── scripts/                      Figure, website, and PDF builders
├── src/
│   ├── data.py                   Data preparation and past-only features
│   ├── experiment.py              Training, selection, evaluation, saving
│   ├── models.py                 K-means model classes
│   └── plots.py                  Plotting helpers
├── tests/test_pipeline.py        Unit tests
├── demo.py                       Saved-model held-out prediction
├── graphs.html                   Generated results website
├── requirements.txt              Runtime dependencies
└── requirements-notebook.txt     Optional notebook dependencies
```

## Installation and setup

The repository does not declare a `pyproject.toml` Python constraint. The current requirements and project instructions target Python 3.12 or newer; the saved results were generated with Python 3.14.3 and scikit-learn 1.7.2. Use a compatible Python installation.

### macOS/Linux

```bash
cd PES1UG24AM206_ML_Mini_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows Command Prompt

```bat
cd /d D:\ML_Mini_Project
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

The catalogs are already included under `data/raw/`; no separate dataset download is required. To fetch the reference submodule as well, clone with:

```bash
git clone --recurse-submodules <repository-url>
```

## How to run

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Run both experiment protocols:

```bash
python -m src.experiment --protocol both
```

This retrains the models and overwrites saved files under `results/` and `models/`. To keep the checked-in outputs unchanged while testing a chronological run, use:

```bash
python -m src.experiment --protocol chronological --output-dir tmp/chrono
```

Run the held-out demo:

```bash
python demo.py --row 0
python demo.py --interactive
```

Rebuild figures from saved results without retraining:

```bash
python scripts/build_figures.py
```

Rebuild the two-page PDF:

```bash
python scripts/build_writeup.py
```

Rebuild the website:

```bash
python scripts/build_site.py
```

Open [`graphs.html`](graphs.html) in a browser from the project root so its relative figure links work. The submitted PDF is [`output/pdf/PEEVED_Mini_Project_Summary.pdf`](output/pdf/PEEVED_Mini_Project_Summary.pdf), and the presentation is [`output/slides/PEEVED_Mini_Project_Review_Final.pptx`](output/slides/PEEVED_Mini_Project_Review_Final.pptx).

## Reproducibility and outputs

- `results/*/metrics.json`: audit information, model selections, development metrics, test metrics, and warnings
- `results/*/test_predictions.csv`: frozen test predictions
- `results/*/tuning.csv`: candidate development records
- `results/*/prepared_data.csv`: prepared rows and split assignments
- `models/`: serialized model bundles used by the demo
- `output/figures/`: generated figures
- `graphs.html`: generated results page
- `output/pdf/`: two-page report
- `output/slides/`: presentation

The current test suite has **17 passing tests**. This count should be rechecked after any code or dependency change.

## Limitations and future work

The chronological test period contains only 19 eruptive examples, so recall, precision, and FPR estimates have wide uncertainty. Later-period ranking performance is close to chance for the main classifiers, and the K-means configuration search improved development AUROC without improving chronological test AUROC. Earthquake-only inputs also omit important volcanic signals such as ground deformation, gas emissions, thermal observations, and seismic waveform information. Future work could add those signals, use longer catalogs, and evaluate uncertainty across multiple time splits without using the final test period for tuning.

## References

- Ben Mullet, Ankush Singh, and Ying-Qi Wong, “Predicting Eruptive Events at Volcanoes from Earthquake Data,” Stanford CS229 project report, Fall 2019.
- Original implementation and catalogs: [bmullet/PEEVED](https://github.com/bmullet/PEEVED)
- Local reference copy: [`reference/PEEVED/`](reference/PEEVED/)

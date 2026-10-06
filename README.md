# PEEVED: eruption activity from earthquake data

**UE24CS352A Machine Learning mini-project** — Prithvish Adhikari (PES1UG24AM206) and Piyush Shiv (PES1UG24AM191)

This mini-project adapts two tasks from Mullet, Singh, and Wong's CS229 study of Pu'u O'o, Kilauea: classify whether an earthquake occurred during an eruptive episode, and estimate days to the next eruption. The source paper reports promising results for a random forest using a random split. This adaptation uses the same bundled catalogs and feature set, applies documented data fixes, and adds a time-ordered evaluation so we can measure performance on later earthquakes without tuning on the test period.

## Set up

Use Python 3.12. In **Command Prompt (cmd.exe)**, create and activate the virtual environment, then install the pinned dependencies:

```bat
cd /d D:\ML_Mini_Project
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
where python
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

After activation, `where python` should list `D:\ML_Mini_Project\.venv\Scripts\python.exe` first and the prompt should start with `(.venv)`. The PowerShell activation command is different: run `.\.venv\Scripts\Activate.ps1` in PowerShell. If you do not want to activate the environment, use `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m src.experiment --protocol both` directly in Command Prompt.

## Run the project

Check the pipeline first (about one second):

```bat
python -m unittest discover -s tests -v
```

To view the saved graphs without running Python, open `graphs.html` in a browser. Keep it in the project folder so its links to `output/figures/` work. It shows the headline metrics, all charts with their tables, the random-split comparison and the key code. Click a graph to open it at full size; the combined chart sheet is `output/figures/kilauea_panel_figures.pdf`.

Run both the chronological and random-split comparisons (about 30 seconds). This overwrites the saved files in `results/` and `models/`; with the pinned versions in `requirements.txt` it reproduces them. To rerun without touching the saved files, write to a separate folder, for example `python -m src.experiment --protocol chronological --output-dir tmp/chrono`.

```bat
python -m src.experiment --protocol both
```

The chronological run writes prepared data, model selection records, test predictions, metrics, and fitted model bundles under `results/chronological/` and `models/`. The random split writes equivalent evaluation files under `results/random/`. Run a held-out prediction from the saved chronological models:

```bat
python demo.py --row 0
```

Use `python demo.py --interactive` to inspect more held-out rows. The demo displays the model score and the catalog labels for comparison. It is a historical classroom demonstration, not a calibrated probability or a volcano warning system.

Build presentation figures from the saved chronological test predictions:

```bat
python scripts/build_figures.py
```

The script reads the saved results (it does not retrain anything) and saves six high-resolution charts plus a combined panel sheet (PDF and PNG) in `output/figures/`: ROC curves, the confusion matrix, regression error, observed versus predicted time to eruption, a chronological-versus-random comparison, and a timeline of the split. Model names and the development-selected models are read from `results/`. Fonts come from `assets/fonts/`, so the scripts run on Windows, macOS and Linux.

Rebuild the two-page PDF summary after rerunning the experiment and the figures (it embeds two charts from `output/figures/`):

```bat
python scripts/build_writeup.py
```

## Notebook walkthrough

`notebooks/walkthrough.ipynb` walks through the data audit, features, labels, the time-ordered split, the models and every chart, with the outputs already saved. It reads the saved results; one optional cell reruns the full experiment into `tmp/` so the saved files are not changed. To open it:

```bat
python -m pip install -r requirements-notebook.txt
python -m jupyter lab notebooks/walkthrough.ipynb
```

## Data and features

The bundled CSV files are the earthquake and Pu'u O'o eruption catalogs used by the reference PEEVED project (byte-identical copies). The paper sources the earthquakes from the ANSS catalog hosted by WOVOdat and the eruption history from the Hawaii Center for Volcanology. `src/data.py` filters earthquakes to the East Rift Zone, removes invalid/duplicate records, and builds seven features: longitude, latitude, depth, magnitude, and earthquake counts in the preceding 1, 7, and 30 days. Rolling counts exclude events at the current timestamp. Eruptive labels include a 24-hour uncertainty buffer. Events before the catalog warm-up period, after the episode-48 coverage cutoff, or without a known next eruption are excluded.

The eruption catalog contains a duration typo for episode 48 (`173566800 hrs`). We retain the episode's known start as an exclusive coverage cutoff and do not use the corrupted duration. The catalog also lists events 35.5 and 35 out of date order; preprocessing sorts onset dates and their matching durations together.

On the 2,960 modeled rows, the eruptive labels match the original PEEVED `was_erupting` rule exactly. Applied to all 3,764 East Rift earthquakes, that rule labels 22.4% as erupting (the paper reports about 20%). The lower 10.3% here comes from two documented exclusions: 282 events after 20 July 1986 that are labelled erupting only through episode 48's corrupted duration, and the 522 earthquakes of the 30-day warm-up in January 1983.

## Evaluation

The primary split is chronological: 70% train, 20% development, and 10% test by time, with a 30-day gap and labels that must have become observable before the next split. Model choices are selected on development data. Classification reports AUROC, average precision, Cohen's kappa, precision, recall, F1, and a confusion matrix. Regression reports RMSE, MAE, and R-squared. The random split is included as a comparison because neighboring events can make a random holdout look more favorable than forecasting later periods.

On the bundled data, 2,960 earthquakes remain for modeling and 10.3% carry an erupting label. Chronological test AUROC is 0.524 for the development-selected class-conditional K-means model; its kappa is -0.012. The development-selected regression baseline has test RMSE 7.75 days and R-squared -0.179. The chronological test period contains only 19 eruptive earthquakes, so these estimates are uncertain (a stratified bootstrap 95% interval for the AUROC is roughly 0.37–0.68, and it is optimistic because nearby earthquakes are correlated). These results do not support reliable eruption forecasting from these seven catalog features alone, and the scores are not calibrated hazard probabilities.

For comparison, the random split (stratified 70/20/10, as in the paper) selects random forests: test AUROC 0.682 and kappa 0.114 for classification, and RMSE 11.01 days with R-squared 0.160 for regression. The paper's random forest reports test kappa 0.52 and AUROC 0.71 (its text quotes kappa 0.42) and R-squared 0.38; its RMSE is on a standardized target and is not comparable with RMSE in days.

## Project files

- `src/data.py`: source cleaning, leakage-aware features, eruption labels
- `src/models.py`: K-means model wrappers
- `src/plots.py`: shared chart style and drawing functions (scripts, notebook)
- `notebooks/walkthrough.ipynb`: narrated walkthrough with saved outputs
- `assets/fonts/`: Open Sans and Montserrat (SIL Open Font License) used by the charts and PDF
- `src/experiment.py`: model selection, chronological/random evaluation, saved outputs
- `demo.py`: held-out prediction demonstration
- `tests/test_pipeline.py`: focused checks for feature and label boundaries
- `output/figures/`: charts built by `scripts/build_figures.py`
- `output/pdf/`: two-page assignment write-up
- `output/slides/`: review presentation
- `graphs.html`: results page built by `scripts/build_site.py` (open it in a browser from the project folder)

`reference/PEEVED` is a Git submodule pointing to the original repository; use `git clone --recurse-submodules` to fetch it. Nothing in the pipeline needs it. Local environments, temporary previews, and build files are excluded from Git.

## Sources

- Ben Mullet, Ankush Singh, and Ying-Qi Wong, “Predicting Eruptive Events at Volcanoes from Earthquake Data,” Stanford CS229 project report, Fall 2019. See `reference/PEEVED/`.
- Original code and bundled catalog source: [bmullet/PEEVED](https://github.com/bmullet/PEEVED).

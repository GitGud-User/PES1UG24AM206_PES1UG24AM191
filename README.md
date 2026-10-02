# PEEVED: eruption activity from earthquake data

This mini-project reproduces two tasks from Mullet, Singh, and Wong's CS229 study of Pu'u O'o, Kilauea: classify whether an earthquake occurred during an eruptive episode, and estimate days to the next eruption. The source paper reports promising results for a random forest. This implementation adds a time-ordered evaluation so we can measure performance on later earthquakes without tuning on the test period.

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

To view the saved graphs without running Python, open `graphs.html` in a browser. Keep it in the project folder so its links to `output/figures/` work. Click a graph for a larger view or open the combined PDF from the page.

Run both the chronological and random-split comparisons:

```bat
python -m src.experiment --protocol both
```

The chronological run writes prepared data, model selection records, test predictions, metrics, and fitted model bundles under `results/chronological/` and `models/`. The random split writes equivalent evaluation files under `results/random/`. Run a held-out prediction from the saved chronological models:

```bat
python demo.py --row 0
```

Use `python demo.py --interactive` to inspect more held-out rows. The demo displays the model score and the catalog labels for comparison. It is a historical classroom demonstration, not a calibrated probability or a volcano warning system.

Rebuild the two-page PDF summary after rerunning the experiment:

```powershell
python scripts/build_writeup.py
```

Build presentation figures from the saved chronological test predictions:

```bat
python scripts/build_figures.py
```

The script saves four individual, high-resolution charts and a combined one-page panel sheet in `output/figures/`. All figures use the chronological holdout; the regression timing plot labels the selected mean baseline and the forest candidate separately.

## Data and features

The bundled CSV files are the earthquake and Pu'u O'o eruption catalogs used by the reference PEEVED project. `src/data.py` filters earthquakes to the East Rift Zone, removes invalid/duplicate records, and builds seven features: longitude, latitude, depth, magnitude, and earthquake counts in the preceding 1, 7, and 30 days. Rolling counts exclude events at the current timestamp. Eruptive labels include a 24-hour uncertainty buffer. Events before the catalog warm-up period, after the episode-48 coverage cutoff, or without a known next eruption are excluded.

The eruption catalog contains a duration typo for episode 48 (`173566800 hrs`). We retain the episode's known start as an exclusive coverage cutoff and do not use the corrupted duration. The catalog also lists events 35.5 and 35 out of date order; preprocessing sorts onset dates and their matching durations together.

## Evaluation

The primary split is chronological: 70% train, 20% development, and 10% test by time, with a 30-day gap and labels that must have become observable before the next split. Model choices are selected on development data. Classification reports AUROC, average precision, Cohen's kappa, precision, recall, F1, and a confusion matrix. Regression reports RMSE, MAE, and R-squared. The random split is included as a comparison because neighboring events can make a random holdout look more favorable than forecasting later periods.

On the bundled data, 2,960 earthquakes remain for modeling and 10.3% carry an erupting label. Chronological test AUROC is 0.524 for the development-selected class-conditional K-means model; its kappa is -0.012. The development-selected regression baseline has test RMSE 7.75 days and R-squared -0.179. These results do not support reliable eruption forecasting from these seven catalog features alone.

## Project files

- `src/data.py`: source cleaning, leakage-aware features, eruption labels
- `src/models.py`: K-means model wrappers
- `src/experiment.py`: model selection, chronological/random evaluation, saved outputs
- `demo.py`: held-out prediction demonstration
- `tests/test_pipeline.py`: focused checks for feature and label boundaries
- `output/pdf/`: two-page assignment write-up
- `output/slides/`: review presentation
- `graphs.html`: browser page for the saved graphs

Local environments, temporary previews, and build files are excluded from Git. The `.build/` folder holds local presentation-generation tooling; it is not needed to run the ML experiment or view the final presentation.

## Sources

- Ben Mullet, Ankush Singh, and Ying-Qi Wong, “Predicting Eruptive Events at Volcanoes from Earthquake Data,” Stanford CS229, Fall 2019. See `reference/PEEVED/` and the supplied paper `26640532.pdf`.
- Original code and bundled catalog source: [bmullet/PEEVED](https://github.com/bmullet/PEEVED).

The assignment asks for the final source in a private GitHub repository shared with faculty and any assigned TAs. This working directory contains the ready-to-run project; publish it to the team's private repository and grant access to the assigned reviewers before submission.

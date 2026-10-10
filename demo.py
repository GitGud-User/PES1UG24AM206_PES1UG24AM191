"""Live demo: an actual saved-model prediction on a held-out earthquake."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from src.data import FEATURES, ROOT


def predict_row(row_index=0):
    result = ROOT/'results/chronological'
    metadata = json.loads((result/'metrics.json').read_text())
    test = pd.read_csv(result/'prepared_data.csv').query("split == 'test'").reset_index(drop=True)
    if not 0 <= row_index < len(test):
        raise ValueError(f'Row must be between 0 and {len(test)-1}')
    row = test.iloc[[row_index]]
    print(f'\nHeld-out earthquake #{row_index} | {row.time.iloc[0]}')
    print(row[FEATURES].to_string(index=False))
    classification_runs = [('joint-F1', metadata['selection']['classification'],
                            ROOT / 'models' / 'classification_selected_joint.joblib')]
    for strategy, selection in metadata.get('selection_kmeans_controlled', {}).items():
        classification_runs.append(
            (f'kmeans-{strategy}', selection,
             ROOT / 'models' / f'classification_kmeans_{strategy}.joblib')
        )
    for strategy, selection in metadata.get(
        'selection_kmeans_improvement', {}
    ).get('operating_points', {}).items():
        classification_runs.append(
            (f'kmeans-improved-{strategy}', selection,
             ROOT / 'models' / f'classification_kmeans_improved_{strategy}.joblib')
        )
    for strategy, selection in metadata.get('selection_recall_first', {}).items():
        if selection.get('status') == 'no_eligible_candidate':
            print(f'\n{strategy}: no eligible candidate under the development FPR cap')
        else:
            classification_runs.append(
                (strategy, selection, ROOT / 'models' / f'classification_{strategy}.joblib')
            )
    for label, selection, path in classification_runs:
        bundle = joblib.load(path)
        model = bundle['model']
        features = bundle['features']
        score = float(model.predict_proba(row[features])[0, 1])
        threshold = float(bundle.get('threshold', .5))
        print(f'\n{label} ({selection["family"]}) features: {", ".join(features)}')
        print(f'{label}: eruption score {score:.3f}; class {int(score >= threshold)} '
              f'(threshold {threshold:.3f})')
    selection = metadata['selection']['regression']
    family = selection['family']
    bundle = joblib.load(ROOT / 'models' / f'regression_{family}.joblib')
    model = bundle['model']
    features = bundle['features']
    days = float(np.maximum(0, model.predict(row[features]))[0])
    print(f'\n{family} features: {", ".join(features)}')
    print(f'{family}: estimated time to eruption {days:.2f} days')
    print(f'Observed catalog labels: erupting={int(row.erupting.iloc[0])}; time to eruption={row.tte_days.iloc[0]:.2f} days')
    print('Historical educational model; scores are not calibrated hazard probabilities.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--row', type=int, default=0)
    p.add_argument('--interactive', action='store_true')
    args = p.parse_args()
    try:
        predict_row(args.row)
        while args.interactive:
            value = input('\nEnter another test row, or q to quit: ').strip()
            if value.lower() == 'q':
                break
            try:
                predict_row(int(value))
            except ValueError as exc:
                print(exc)
    except (FileNotFoundError, ValueError) as exc:
        p.exit(1, f'{exc}\nRun python -m src.experiment first.\n')

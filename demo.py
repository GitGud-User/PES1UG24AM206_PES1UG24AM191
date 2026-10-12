"""Live demo: an actual saved-model prediction on a held-out earthquake."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from src.data import FEATURES, ROOT


def _section(title):
    print(f'\n{title}')
    print('-' * len(title))


def _print_table(headers, rows, right_align=None):
    right_align = set(right_align or ())
    rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print('  '.join(
        header.rjust(widths[index]) if index in right_align else header.ljust(widths[index])
        for index, header in enumerate(headers)
    ))
    print('  '.join('-' * width for width in widths))
    for row in rows:
        print('  '.join(
            value.rjust(widths[index]) if index in right_align else value.ljust(widths[index])
            for index, value in enumerate(row)
        ))


def _metric(value, digits=3):
    return f'{value:.{digits}f}'


def predict_row(row_index=0):
    result = ROOT/'results/chronological'
    metadata = json.loads((result/'metrics.json').read_text())
    test = pd.read_csv(result/'prepared_data.csv').query("split == 'test'").reset_index(drop=True)
    if not 0 <= row_index < len(test):
        raise ValueError(f'Row must be between 0 and {len(test)-1}')
    row = test.iloc[[row_index]]
    print('\nVolcano Eruption Prediction')
    print('===========================')
    print(f'Selected held-out earthquake row: {row_index}')
    print(f'Event timestamp: {row.time.iloc[0]}')
    _section('Input features')
    _print_table(
        ['Feature', 'Value'],
        [[feature, f'{row[feature].iloc[0]:.6g}'] for feature in FEATURES],
        right_align={1},
    )
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
    classification_rows = []
    for label, selection, path in classification_runs:
        bundle = joblib.load(path)
        model = bundle['model']
        features = bundle['features']
        score = float(model.predict_proba(row[features])[0, 1])
        threshold = float(bundle.get('threshold', .5))
        strategy = 'Historical K-means' if label.startswith('kmeans-') and not label.startswith(
            'kmeans-improved-'
        ) else 'Improved K-means search' if label.startswith('kmeans-improved-') else selection['family']
        classification_rows.append([
            label,
            strategy,
            _metric(score),
            _metric(threshold),
            f'{("Eruptive" if score >= threshold else "Not eruptive")} ({int(score >= threshold)})',
        ])
    _section('Classification predictions')
    _print_table(
        ['Model', 'Strategy', 'Score', 'Threshold', 'Predicted class'],
        classification_rows, right_align={2, 3},
    )
    print('Scores are model outputs, not calibrated probabilities.')

    selection = metadata['selection']['regression']
    family = selection['family']
    bundle = joblib.load(ROOT / 'models' / f'regression_{family}.joblib')
    model = bundle['model']
    features = bundle['features']
    days = float(np.maximum(0, model.predict(row[features]))[0])
    _section('Regression prediction')
    _print_table(
        ['Quantity', 'Value'],
        [
            ['Predicted days until eruption', f'{days:.2f} days ({family})'],
            ['Observed catalog label', f'erupting={int(row.erupting.iloc[0])}'],
            ['Observed catalog time-to-eruption', f'{row.tte_days.iloc[0]:.2f} days'],
        ],
    )
    print('Disclaimer: historical educational models; not calibrated hazard probabilities or an operational warning system.')


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

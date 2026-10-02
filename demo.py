"""Live demo: an actual saved-model prediction on a held-out earthquake."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from src.data import ROOT, FEATURES


def predict_row(row_index=0):
    result = ROOT/'results/chronological'
    metadata = json.loads((result/'metrics.json').read_text())
    test = pd.read_csv(result/'prepared_data.csv').query("split == 'test'").reset_index(drop=True)
    if not 0 <= row_index < len(test):
        raise ValueError(f'Row must be between 0 and {len(test)-1}')
    row = test.iloc[[row_index]]
    print(f'\nHeld-out earthquake #{row_index} | {row.time.iloc[0]}')
    print(row[FEATURES].to_string(index=False))
    for task in ('classification','regression'):
        family = metadata['selection'][task]['family']
        bundle = joblib.load(ROOT/'models'/f'{task}_{family}.joblib')
        model = bundle['model']
        if task == 'classification':
            score = float(model.predict_proba(row[FEATURES])[0,1])
            print(f'\n{family}: eruption score {score:.3f}; class {int(score >= .5)} (threshold 0.5)')
        else:
            days = float(np.maximum(0, model.predict(row[FEATURES]))[0])
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

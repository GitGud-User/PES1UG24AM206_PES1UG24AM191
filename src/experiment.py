"""Run development selection and held-out evaluation; never tune on test results."""
import argparse
import json
import platform
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, cohen_kappa_score, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import FEATURES, ROOT, prepare
from .models import ClassConditionalKMeans, JointKMeansRegressor


def split_data(data, protocol, seed=42, gap_days=30):
    indices = np.arange(len(data))
    if protocol == 'random':
        train, rest = train_test_split(indices, test_size=.3, stratify=data.erupting, random_state=seed)
        dev, test = train_test_split(rest, test_size=1/3, stratify=data.erupting.iloc[rest], random_state=seed)
    else:
        unique = data.time.drop_duplicates().sort_values().to_numpy()
        dev_start, test_start = pd.Timestamp(unique[int(.7*len(unique))]), pd.Timestamp(unique[int(.9*len(unique))])
        gap = pd.Timedelta(days=gap_days)
        train = indices[(data.time < dev_start-gap) & (data.label_available_time < dev_start)]
        dev = indices[(data.time >= dev_start) & (data.time < test_start-gap) & (data.label_available_time < test_start)]
        test = indices[data.time >= test_start]
    if any(len(x) == 0 for x in (train, dev, test)):
        raise ValueError('A split is empty')
    for ids in (train, dev, test):
        if data.erupting.iloc[ids].nunique() != 2:
            raise ValueError('Both classes must occur in every split')
    return {'train': np.sort(train), 'dev': np.sort(dev), 'test': np.sort(test)}


def candidates(task, seed):
    if task == 'classification':
        yield 'baseline', 'prior', DummyClassifier(strategy='prior')
        for c in (.1, 1, 10):
            yield 'logistic', f'C={c}', make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=seed))
        for depth in (5, 15, None):
            yield 'forest', f'depth={depth}', RandomForestClassifier(n_estimators=200, max_depth=depth, min_samples_leaf=2, max_features=3, class_weight='balanced', random_state=seed, n_jobs=1)
        for k in (4, 8):
            yield 'kmeans', f'k={k}', make_pipeline(StandardScaler(), ClassConditionalKMeans(k, seed))
        for alpha in (.001, .1):
            yield 'mlp', f'alpha={alpha}', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,16), alpha=alpha, max_iter=400, random_state=seed, early_stopping=False))
    else:
        yield 'baseline', 'mean', DummyRegressor(strategy='mean')
        for depth in (5, 15, None):
            yield 'forest', f'depth={depth}', RandomForestRegressor(n_estimators=200, max_depth=depth, min_samples_leaf=2, max_features=3, random_state=seed, n_jobs=1)
        for k in (6, 10):
            yield 'kmeans', f'k={k}', TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), JointKMeansRegressor(k, seed)), transformer=StandardScaler())
        for alpha in (.001, .1):
            yield 'mlp', f'alpha={alpha}', TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,16), alpha=alpha, max_iter=400, random_state=seed, early_stopping=False)), transformer=StandardScaler())


def classification_metrics(y, score):
    pred = (score >= .5).astype(int)
    return dict(auroc=float(roc_auc_score(y, score)), average_precision=float(average_precision_score(y, score)),
                kappa=float(cohen_kappa_score(y, pred)), accuracy=float(accuracy_score(y, pred)),
                precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)), confusion_matrix=confusion_matrix(y, pred, labels=[0,1]).tolist())


def regression_metrics(y, pred):
    return dict(rmse_days=float(np.sqrt(mean_squared_error(y, pred))), mae_days=float(mean_absolute_error(y, pred)), r2=float(r2_score(y, pred)))


def run(protocol='chronological', seed=42, buffer_hours=24, output_dir=None):
    out = Path(output_dir or ROOT / 'results' / protocol)
    out.mkdir(parents=True, exist_ok=True)
    data, audit = prepare(buffer_hours=buffer_hours)
    split = split_data(data, protocol, seed)
    data['split'] = 'excluded_gap_or_label_horizon'
    for name, ids in split.items():
        data.loc[ids, 'split'] = name
    data.to_csv(out/'prepared_data.csv', index=False)
    audit['splits'] = {name: {'rows': len(ids), 'positive_rows': int(data.erupting.iloc[ids].sum()),
        'start': str(data.time.iloc[ids].min()), 'end': str(data.time.iloc[ids].max())} for name,ids in split.items()}
    audit['excluded_split_rows'] = int((data.split == 'excluded_gap_or_label_horizon').sum())
    summary = {'protocol': protocol, 'seed': seed, 'audit': audit, 'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
               'selection': {}, 'metrics': {}, 'warnings': []}
    X = data[FEATURES]
    train, dev, test = (split[k] for k in ('train','dev','test'))
    tuning = []
    predictions = data.iloc[test][['source_row','time','erupting','tte_days']].copy()
    for task in ('classification','regression'):
        y = data['erupting' if task == 'classification' else 'tte_days']
        best = {}
        for family, config, model in candidates(task, seed):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(X.iloc[train], y.iloc[train])
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    summary['warnings'].append(f'{task}/{family}/{config}: {w.message}')
            pred = model.predict_proba(X.iloc[dev])[:,1] if task == 'classification' else np.maximum(0, model.predict(X.iloc[dev]))
            metric = classification_metrics(y.iloc[dev], pred) if task == 'classification' else regression_metrics(y.iloc[dev], pred)
            value = metric['auroc'] if task == 'classification' else -metric['rmse_days']
            tuning.append({'task':task,'family':family,'config':config,**{k:v for k,v in metric.items() if k!='confusion_matrix'}})
            if family not in best or value > best[family][0]:
                best[family] = (value, config, model, metric)
        # Winner is frozen before computing any test metrics; baseline may win.
        winner = max(best, key=lambda f: best[f][0])
        summary['selection'][task] = {'family':winner,'config':best[winner][1], 'development_metrics':best[winner][3]}
        summary['metrics'][task] = {}
        for family, (_, config, model, dev_metrics) in best.items():
            pred = model.predict_proba(X.iloc[test])[:,1] if task == 'classification' else np.maximum(0, model.predict(X.iloc[test]))
            metrics = classification_metrics(y.iloc[test], pred) if task == 'classification' else regression_metrics(y.iloc[test], pred)
            if task == 'regression':
                repose = data.erupting.iloc[test].to_numpy() == 0
                metrics['repose_only'] = regression_metrics(y.iloc[test].to_numpy()[repose], pred[repose])
            summary['metrics'][task][family] = {'config':config,'development':dev_metrics,'test':metrics}
            predictions[f'{task}_{family}'] = pred
            if protocol == 'chronological' and output_dir is None:
                model_dir = ROOT/'models'
                model_dir.mkdir(exist_ok=True)
                joblib.dump({'model': model, 'features':FEATURES,'task':task,'family':family,'config':config,'seed':seed,'buffer_hours':buffer_hours}, model_dir/f'{task}_{family}.joblib', compress=3)
        print(f'{protocol} {task}: dev-selected {winner}; test {summary["metrics"][task][winner]["test"]}', flush=True)
    predictions.to_csv(out/'test_predictions.csv', index=False)
    pd.DataFrame(tuning).to_csv(out/'tuning.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=['chronological','random','both'], default='both')
    parser.add_argument('--buffer-hours', type=float, default=24)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.buffer_hours < 0:
        parser.error('--buffer-hours must be nonnegative')
    if args.protocol == 'both' and args.output_dir:
        parser.error('Use a single protocol with --output-dir')
    with threadpool_limits(limits=1):
        for protocol in (['chronological','random'] if args.protocol == 'both' else [args.protocol]):
            run(protocol, buffer_hours=args.buffer_hours, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

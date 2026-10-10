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
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score,
    r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import ALL_FEATURES, FEATURES, ROOT, prepare
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
        yield 'extra_trees', 'n=200,depth=None', ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, max_features=3, class_weight='balanced', random_state=seed, n_jobs=1)
        yield 'gradient_boosting', 'n=100,depth=2', GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=.05, random_state=seed)
        for k in (4, 8):
            yield 'kmeans', f'k={k}', make_pipeline(StandardScaler(), ClassConditionalKMeans(k, seed))
        for alpha in (.001, .1):
            yield 'mlp', f'alpha={alpha}', make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(32, 16), alpha=alpha, max_iter=1000,
                              random_state=seed, early_stopping=True, validation_fraction=.15,
                              n_iter_no_change=30),
            )
    else:
        yield 'baseline', 'mean', DummyRegressor(strategy='mean')
        for depth in (5, 15, None):
            yield 'forest', f'depth={depth}', RandomForestRegressor(n_estimators=200, max_depth=depth, min_samples_leaf=2, max_features=3, random_state=seed, n_jobs=1)
        for k in (6, 10):
            yield 'kmeans', f'k={k}', TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), JointKMeansRegressor(k, seed)), transformer=StandardScaler())
        for alpha in (.001, .1):
            yield 'mlp', f'alpha={alpha}', TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(32, 16), alpha=alpha, max_iter=1000,
                                 random_state=seed, early_stopping=True, validation_fraction=.15,
                                 n_iter_no_change=30),
                ),
                transformer=StandardScaler(),
            )


def kmeans_improvement_candidates(seed):
    """Controlled K-means grid; every scaler is fitted inside its pipeline."""
    for k in (2, 3, 4, 5, 6, 8):
        for scaling in ('raw', 'standard'):
            for score_mode in ('relative_distance', 'normalized_margin'):
                if scaling == 'standard':
                    model = make_pipeline(
                        StandardScaler(),
                        ClassConditionalKMeans(k, seed, score_mode),
                    )
                else:
                    model = ClassConditionalKMeans(k, seed, score_mode)
                yield {
                    'family': 'kmeans',
                    'config': f'k={k},scaling={scaling},score={score_mode}',
                    'feature_set': 'original_7',
                    'k': k,
                    'scaling': scaling,
                    'score_mode': score_mode,
                    'model': model,
                }


def _classification_at_threshold(y, score, threshold):
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    negative_count = tn + fp
    return dict(
        threshold=float(threshold),
        kappa=float(cohen_kappa_score(y, pred)),
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        false_positive_rate=float(fp / negative_count) if negative_count else 0.0,
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )


def select_classification_threshold(y, score):
    """Select an F1-maximizing threshold on development data only.

    Tiny or one-class development samples use 0.5 because threshold
    optimization is not identifiable in that case.
    """
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError('Labels and finite scores must be aligned one-dimensional arrays')
    if len(np.unique(y)) < 2 or min(np.bincount(y, minlength=2)) < 2:
        return .5, 'fixed_0.5_insufficient_class_support'
    # Include a threshold above every score so an all-negative operating
    # point can be evaluated and selected when it is the only cap-safe choice.
    thresholds = classification_thresholds(y, score)
    evaluations = [(_classification_at_threshold(y, score, t), float(t)) for t in thresholds]
    # Equal-F1 thresholds use the largest threshold, a deterministic rule that
    # prefers fewer positive calls; score ties are already collapsed above.
    best = max(evaluations, key=lambda item: (item[0]['f1'], item[1]))
    return best[1], 'development_f1'


def classification_thresholds(y, score):
    """Return deterministic development-only thresholds, including all-negative."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError('Labels and finite scores must be aligned one-dimensional arrays')
    if not len(score):
        raise ValueError('At least one score is required')
    return np.unique(np.r_[.5, score, np.nextafter(score.max(), np.inf)])


def select_kmeans_controlled(record, y_dev, dev_score):
    """Select fixed-K-means operating points from development scores only."""
    thresholds = classification_thresholds(y_dev, dev_score)
    threshold_records = []
    for threshold in thresholds:
        metrics = classification_metrics(y_dev, dev_score, threshold)
        threshold_records.append({
            **record,
            'threshold': float(threshold),
            'development_metrics': metrics,
        })

    f1_record = max(
        threshold_records,
        key=lambda candidate: (
            candidate['development_metrics']['f1'],
            candidate['threshold'],
        ),
    )
    recall_records = {}
    for cap in (.10, .20):
        recall_records[f'recall_first_fpr_{cap:.2f}'.replace('.', '_')] = (
            select_recall_first_threshold(threshold_records, cap)
        )
    fixed = next(
        candidate for candidate in threshold_records
        if candidate['threshold'] == .5
    )
    return {
        'original_fixed_0_5': fixed,
        'development_f1': f1_record,
        **recall_records,
    }


def select_kmeans_improvement_candidate(records):
    """Select a K-means ranking configuration using development AUROC then AP."""
    if not records:
        raise ValueError('At least one K-means improvement candidate is required')
    return max(
        records,
        key=lambda record: (
            record['development_metrics']['auroc'],
            record['development_metrics']['average_precision'],
            record['config'],
        ),
    )


def select_recall_first_threshold(records, fpr_cap):
    """Select a threshold for one score stream using the K-means tie rule."""
    if not records:
        raise ValueError('At least one threshold evaluation is required')
    eligible = [
        record for record in records
        if np.isfinite(record['development_metrics']['false_positive_rate'])
        and record['development_metrics']['false_positive_rate'] <= fpr_cap
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda record: (
            record['development_metrics']['recall'],
            -record['development_metrics']['false_positives'],
            record['development_metrics']['precision'],
            record['threshold'],
        ),
    )


def select_recall_first_candidate(records, fpr_cap):
    """Select the highest-recall development operating point under an FPR cap.

    Candidates are eligible only when their development false-positive rate is
    finite and no greater than ``fpr_cap``. Ties are resolved by fewer false
    positives, higher precision, higher average precision, higher threshold,
    then feature set, family, and configuration names.
    """
    if not 0 <= fpr_cap <= 1:
        raise ValueError('fpr_cap must be between 0 and 1')
    if not records:
        raise ValueError('At least one classification candidate is required')
    eligible = [
        record for record in records
        if np.isfinite(record['development_metrics']['false_positive_rate'])
        and record['development_metrics']['false_positive_rate'] <= fpr_cap
    ]
    if not eligible:
        return None

    def key(record):
        metric = record['development_metrics']
        return (
            metric['recall'],
            -metric['false_positives'],
            metric['precision'],
            metric['average_precision'],
            record['threshold'],
            record['feature_set'],
            record['family'],
            record['config'],
        )

    return max(eligible, key=key)


def select_classification_candidate(records, objective='f1'):
    """Select a fitted classification candidate from development metrics only.

    F1 is the primary objective for the new procedure, followed by average
    precision. The final deterministic tie-breakers are higher threshold,
    feature-set name, family name, and configuration name. The ``auroc``
    objective reproduces the prior ranking-based selection baseline.
    """
    if objective not in {'f1', 'auroc'}:
        raise ValueError("objective must be 'f1' or 'auroc'")
    if not records:
        raise ValueError('At least one classification candidate is required')

    def key(record):
        metric = record['development_metrics']
        primary = metric['f1'] if objective == 'f1' else metric['auroc']
        return (
            primary,
            metric['average_precision'],
            metric['threshold'],
            record['feature_set'],
            record['family'],
            record['config'],
        )

    return max(records, key=key)


def classification_metrics(y, score, threshold=.5):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.ndim != 1 or score.ndim != 1 or len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError('Labels and finite scores must be aligned one-dimensional arrays')
    if len(np.unique(y)) < 2:
        ranking = {'auroc': float('nan'), 'average_precision': float(average_precision_score(y, score))}
    else:
        ranking = dict(auroc=float(roc_auc_score(y, score)),
                       average_precision=float(average_precision_score(y, score)))
    operating = _classification_at_threshold(y, score, threshold)
    baseline = _classification_at_threshold(y, score, .5)
    return {**ranking, **operating, 'threshold_0_5': baseline}


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
               'selection': {}, 'selection_baseline': {}, 'metrics': {}, 'warnings': []}
    X_by_set = {'original_7': data[FEATURES], 'expanded_11': data[ALL_FEATURES]}
    train, dev, test = (split[k] for k in ('train','dev','test'))
    tuning = []
    predictions = data.iloc[test][['source_row','time','erupting','tte_days']].copy()
    # Classification selection evaluates every feature/family/configuration
    # on development labels, then freezes both model and threshold.
    y = data['erupting']
    classification_records = []
    for feature_set, X in X_by_set.items():
        for family, config, model in candidates('classification', seed):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(X.iloc[train], y.iloc[train])
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    summary['warnings'].append(f'classification/{family}/{config}: {w.message}')
            dev_score = model.predict_proba(X.iloc[dev])[:, 1]
            threshold, threshold_method = select_classification_threshold(y.iloc[dev], dev_score)
            dev_metrics = classification_metrics(y.iloc[dev], dev_score, threshold)
            dev_metrics['threshold_method'] = threshold_method
            record = {'family': family, 'config': config, 'feature_set': feature_set,
                      'model': model, 'development_metrics': dev_metrics,
                      'dev_score': dev_score,
                      'threshold': threshold, 'X': X}
            classification_records.append(record)
            tuning.append({'task': 'classification', 'feature_set': feature_set,
                           'family': family, 'config': config,
                           **{k: v for k, v in dev_metrics.items()
                              if k not in ('confusion_matrix', 'threshold_0_5')}})

    # The old procedure is retained as a separately labeled AUROC baseline.
    baseline_record = select_classification_candidate(classification_records, 'auroc')
    selected_record = select_classification_candidate(classification_records, 'f1')
    recall_records = {
        'recall_first_fpr_0_10': select_recall_first_candidate(classification_records, .10),
        'recall_first_fpr_0_20': select_recall_first_candidate(classification_records, .20),
    }
    controlled_candidates = [
        record for record in classification_records
        if record['family'] == 'kmeans'
        and record['feature_set'] == 'original_7'
        and record['config'] == 'k=4'
    ]
    if len(controlled_candidates) != 1:
        raise RuntimeError('Expected exactly one original-7 K-means k=4 candidate')
    controlled_kmeans = select_kmeans_controlled(
        controlled_candidates[0], y.iloc[dev].to_numpy(),
        controlled_candidates[0]['dev_score'],
    )
    kmeans_improvement_records = []
    for candidate in kmeans_improvement_candidates(seed):
        model = candidate['model']
        model.fit(X_by_set['original_7'].iloc[train], y.iloc[train])
        dev_score = model.predict_proba(X_by_set['original_7'].iloc[dev])[:, 1]
        dev_metrics = classification_metrics(y.iloc[dev], dev_score, .5)
        kmeans_improvement_records.append({
            **candidate,
            'X': X_by_set['original_7'],
            'dev_score': dev_score,
            'threshold': .5,
            'development_metrics': dev_metrics,
        })
        tuning.append({
            'task': 'kmeans_improvement',
            'feature_set': 'original_7',
            'family': 'kmeans',
            'config': candidate['config'],
            **{key: value for key, value in dev_metrics.items()
               if key not in ('confusion_matrix', 'threshold_0_5')},
        })
    kmeans_improvement_winner = select_kmeans_improvement_candidate(
        kmeans_improvement_records
    )
    kmeans_improvement_points = select_kmeans_controlled(
        kmeans_improvement_winner, y.iloc[dev].to_numpy(),
        kmeans_improvement_winner['dev_score'],
    )

    def selection_payload(record):
        return {'family': record['family'], 'config': record['config'],
                'feature_set': record['feature_set'], 'threshold': record['threshold'],
                'development_metrics': record['development_metrics']}

    summary['selection_baseline']['classification'] = selection_payload(baseline_record)
    summary['selection']['classification'] = selection_payload(selected_record)
    summary['selection_recall_first'] = {}
    for strategy, record in recall_records.items():
        summary['selection_recall_first'][strategy] = (
            selection_payload(record) if record is not None
            else {'status': 'no_eligible_candidate', 'fpr_cap': .10 if '0_10' in strategy else .20}
        )
    summary['selection_kmeans_controlled'] = {}
    for strategy, record in controlled_kmeans.items():
        summary['selection_kmeans_controlled'][strategy] = selection_payload(record)
    summary['selection_kmeans_improvement'] = {
        'ranking_model': {
            'family': kmeans_improvement_winner['family'],
            'config': kmeans_improvement_winner['config'],
            'feature_set': 'original_7',
            'k': kmeans_improvement_winner['k'],
            'scaling': kmeans_improvement_winner['scaling'],
            'score_mode': kmeans_improvement_winner['score_mode'],
            'development_metrics': kmeans_improvement_winner['development_metrics'],
        },
        'operating_points': {},
    }
    for strategy, record in kmeans_improvement_points.items():
        summary['selection_kmeans_improvement']['operating_points'][strategy] = (
            selection_payload(record)
        )
    summary['metrics']['classification'] = {}
    for family in sorted({record['family'] for record in classification_records}):
        family_records = [r for r in classification_records if r['family'] == family]
        record = select_classification_candidate(family_records, 'auroc')
        if record['family'] == selected_record['family']:
            record = selected_record
        test_score = record['model'].predict_proba(record['X'].iloc[test])[:, 1]
        test_metrics = classification_metrics(y.iloc[test], test_score, record['threshold'])
        summary['metrics']['classification'][family] = {
            'config': record['config'], 'feature_set': record['feature_set'],
            'threshold': record['threshold'], 'development': record['development_metrics'],
            'test': test_metrics,
        }
        predictions[f'classification_{family}'] = test_score
        if record is selected_record:
            predictions['classification_selected'] = test_score
        if protocol == 'chronological' and output_dir is None:
            model_dir = ROOT / 'models'
            model_dir.mkdir(exist_ok=True)
            joblib.dump({'model': record['model'], 'features': list(record['X'].columns),
                         'task': 'classification', 'family': record['family'],
                         'config': record['config'], 'threshold': record['threshold'],
                         'seed': seed, 'buffer_hours': buffer_hours},
                        model_dir / f'classification_{family}.joblib', compress=3)

    baseline_test_score = baseline_record['model'].predict_proba(
        baseline_record['X'].iloc[test])[:, 1]
    baseline_test_metrics = classification_metrics(
        y.iloc[test], baseline_test_score, baseline_record['threshold'])
    selected_test_score = selected_record['model'].predict_proba(
        selected_record['X'].iloc[test])[:, 1]
    selected_test_metrics = classification_metrics(
        y.iloc[test], selected_test_score, selected_record['threshold'])
    summary['selection_baseline']['classification']['test_metrics'] = baseline_test_metrics
    summary['selection']['classification']['test_metrics'] = selected_test_metrics
    summary['metrics']['classification']['selected_joint'] = {
        'config': selected_record['config'], 'feature_set': selected_record['feature_set'],
        'threshold': selected_record['threshold'],
        'development': selected_record['development_metrics'],
        'test': selected_test_metrics,
    }
    predictions['classification_selected'] = selected_test_score
    if protocol == 'chronological' and output_dir is None:
        joblib.dump({'model': selected_record['model'],
                     'features': list(selected_record['X'].columns),
                     'task': 'classification', 'family': selected_record['family'],
                     'config': selected_record['config'],
                     'threshold': selected_record['threshold'], 'seed': seed,
                     'buffer_hours': buffer_hours},
                    ROOT / 'models' / 'classification_selected_joint.joblib', compress=3)

    for strategy, record in recall_records.items():
        if record is None:
            continue
        score = record['model'].predict_proba(record['X'].iloc[test])[:, 1]
        test_metrics = classification_metrics(y.iloc[test], score, record['threshold'])
        summary['selection_recall_first'][strategy]['test_metrics'] = test_metrics
        predictions[f'classification_{strategy}'] = score
        if protocol == 'chronological' and output_dir is None:
            joblib.dump(
                {'model': record['model'], 'features': list(record['X'].columns),
                 'task': 'classification', 'family': record['family'],
                 'config': record['config'], 'threshold': record['threshold'],
                 'strategy': strategy, 'seed': seed, 'buffer_hours': buffer_hours},
                ROOT / 'models' / f'classification_{strategy}.joblib', compress=3)

    summary['metrics']['kmeans_controlled'] = {}
    for strategy, record in controlled_kmeans.items():
        score = record['model'].predict_proba(record['X'].iloc[test])[:, 1]
        test_metrics = classification_metrics(y.iloc[test], score, record['threshold'])
        summary['selection_kmeans_controlled'][strategy]['test_metrics'] = test_metrics
        summary['metrics']['kmeans_controlled'][strategy] = {
            'config': record['config'], 'feature_set': record['feature_set'],
            'threshold': record['threshold'],
            'development': record['development_metrics'], 'test': test_metrics,
        }
        predictions[f'classification_kmeans_{strategy}'] = score
        if protocol == 'chronological' and output_dir is None:
            joblib.dump(
                {'model': record['model'], 'features': list(record['X'].columns),
                 'task': 'classification', 'family': record['family'],
                 'config': record['config'], 'threshold': record['threshold'],
                 'strategy': f'kmeans_{strategy}', 'seed': seed,
                 'buffer_hours': buffer_hours},
                ROOT / 'models' / f'classification_kmeans_{strategy}.joblib',
                compress=3,
            )

    summary['metrics']['kmeans_improvement'] = {}
    for strategy, record in kmeans_improvement_points.items():
        score = record['model'].predict_proba(record['X'].iloc[test])[:, 1]
        test_metrics = classification_metrics(y.iloc[test], score, record['threshold'])
        summary['selection_kmeans_improvement']['operating_points'][strategy][
            'test_metrics'
        ] = test_metrics
        summary['metrics']['kmeans_improvement'][strategy] = {
            'config': record['config'], 'feature_set': 'original_7',
            'k': record['k'], 'scaling': record['scaling'],
            'score_mode': record['score_mode'], 'threshold': record['threshold'],
            'development': record['development_metrics'], 'test': test_metrics,
        }
        predictions[f'classification_kmeans_improved_{strategy}'] = score
        if protocol == 'chronological' and output_dir is None:
            joblib.dump(
                {'model': record['model'], 'features': list(record['X'].columns),
                 'task': 'classification', 'family': 'kmeans',
                 'config': record['config'], 'threshold': record['threshold'],
                 'strategy': f'kmeans_improved_{strategy}', 'seed': seed,
                 'buffer_hours': buffer_hours},
                ROOT / 'models' / f'classification_kmeans_improved_{strategy}.joblib',
                compress=3,
            )
    print(f'{protocol} classification: joint-dev-F1 selected {selected_record["family"]}; '
          f'test {selected_test_metrics}', flush=True)

    # Regression path is unchanged: select by development RMSE only.
    y = data['tte_days']
    best = {}
    for feature_set, X in [('original_7', X_by_set['original_7'])]:
        for family, config, model in candidates('regression', seed):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(X.iloc[train], y.iloc[train])
            for w in caught:
                if issubclass(w.category, ConvergenceWarning):
                    summary['warnings'].append(f'regression/{family}/{config}: {w.message}')
            dev_pred = np.maximum(0, model.predict(X.iloc[dev]))
            dev_metrics = regression_metrics(y.iloc[dev], dev_pred)
            tuning.append({'task': 'regression', 'feature_set': feature_set,
                           'family': family, 'config': config, **dev_metrics})
            if family not in best or dev_metrics['rmse_days'] < best[family][0]:
                best[family] = (dev_metrics['rmse_days'], config, model, dev_metrics, feature_set)
    winner = min(best, key=lambda family: best[family][0])
    summary['selection']['regression'] = {
        'family': winner, 'config': best[winner][1], 'feature_set': best[winner][4],
        'threshold': None, 'development_metrics': best[winner][3],
    }
    summary['selection_baseline']['regression'] = summary['selection']['regression']
    summary['metrics']['regression'] = {}
    for family, (_, config, model, dev_metrics, feature_set) in best.items():
        X = X_by_set[feature_set]
        pred = np.maximum(0, model.predict(X.iloc[test]))
        metrics = regression_metrics(y.iloc[test], pred)
        repose = data.erupting.iloc[test].to_numpy() == 0
        metrics['repose_only'] = regression_metrics(y.iloc[test].to_numpy()[repose], pred[repose])
        summary['metrics']['regression'][family] = {
            'config': config, 'feature_set': feature_set, 'threshold': None,
            'development': dev_metrics, 'test': metrics,
        }
        predictions[f'regression_{family}'] = pred
        if protocol == 'chronological' and output_dir is None:
            model_dir = ROOT / 'models'
            model_dir.mkdir(exist_ok=True)
            joblib.dump({'model': model, 'features': list(X.columns), 'task': 'regression',
                         'family': family, 'config': config, 'threshold': None,
                         'seed': seed, 'buffer_hours': buffer_hours},
                        model_dir / f'regression_{family}.joblib', compress=3)
    print(f'{protocol} regression: dev-selected {winner}; '
          f'test {summary["metrics"]["regression"][winner]["test"]}', flush=True)
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

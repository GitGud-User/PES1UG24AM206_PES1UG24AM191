import unittest
import numpy as np
import pandas as pd
from src.data import ALL_FEATURES, assign_labels, past_counts, past_temporal_features, prepare, FEATURES
from src.experiment import (classification_metrics, select_classification_candidate,
                            select_classification_threshold, select_kmeans_controlled,
                            kmeans_improvement_candidates,
                            select_kmeans_improvement_candidate,
                            select_recall_first_candidate, split_data)
from src.models import ClassConditionalKMeans, JointKMeansRegressor


class PipelineTests(unittest.TestCase):
    def test_counts_exact_boundaries_and_ties(self):
        t = pd.to_datetime(['2000-01-01','2000-01-02','2000-01-02','2000-01-03 00:00:01'], format='mixed')
        np.testing.assert_array_equal(past_counts(t,1), [0,1,1,0])
        # Appending future events cannot alter past features.
        np.testing.assert_array_equal(past_counts(t,7)[:3], past_counts(t[:3],7))

    def test_extended_features_are_strictly_past_only(self):
        t = pd.to_datetime(['2000-01-01 00:00', '2000-01-01 01:00',
                            '2000-01-01 01:00', '2000-01-02 00:00'])
        m = np.array([2., 4., 9., 8.])
        features = past_temporal_features(t, m)
        np.testing.assert_allclose(features[0], [0, 0, 0, 0])
        np.testing.assert_allclose(features[1], [1, 1, 2, 2])
        np.testing.assert_allclose(features[2], [1, 1, 2, 2])
        np.testing.assert_allclose(features[3], [0, 23, 5, 9])
        np.testing.assert_allclose(features[:3], past_temporal_features(t[:3], m[:3]))

    def test_labels_onset_end_and_unknown(self):
        t = pd.to_datetime(['2000-01-01','2000-01-02','2000-01-02 12:00','2000-01-03','2000-01-05'], format='mixed')
        y, days, _ = assign_labels(t, pd.to_datetime(['2000-01-02','2000-01-04']), [12,12], 0)
        np.testing.assert_array_equal(y, [0,1,0,0,0])
        np.testing.assert_allclose(days[:4], [1,0,1.5,1])
        self.assertTrue(np.isnan(days[-1]))

    def test_buffer_rule(self):
        y, days, _ = assign_labels(pd.to_datetime(['2000-01-03']), pd.to_datetime(['2000-01-02']), [12],24)
        self.assertEqual(y[0],1)
        self.assertEqual(days[0],0)

    def test_data_and_temporal_split(self):
        d,a = prepare()
        self.assertTrue(np.isfinite(d[FEATURES].to_numpy()).all())
        self.assertTrue((d.tte_days >= 0).all())
        self.assertTrue((d.time < pd.Timestamp('1986-07-20')).all())
        s = split_data(d,'chronological')
        self.assertFalse(set(s['train']) & set(s['dev']))
        self.assertFalse(set(s['dev']) & set(s['test']))
        self.assertGreater(d.time.iloc[s['dev']].min() - d.time.iloc[s['train']].max(),pd.Timedelta(days=30))
        self.assertLess(d.label_available_time.iloc[s['train']].max(),d.time.iloc[s['dev']].min())
        self.assertLess(d.label_available_time.iloc[s['dev']].max(),d.time.iloc[s['test']].min())
        self.assertNotIn('tte_days',FEATURES)
        self.assertNotIn('erupting',FEATURES)
        self.assertTrue(set(FEATURES).issubset(ALL_FEATURES))

    def test_centroid_models_use_features_at_inference(self):
        X = np.array([[0,0],[0,.1],[4,4],[4,4.1]])
        c = ClassConditionalKMeans(1).fit(X,[0,0,1,1])
        np.testing.assert_array_equal(c.predict(X),[0,0,1,1])
        r = JointKMeansRegressor(2).fit(X,[0,0,10,10])
        np.testing.assert_allclose(r.predict(X),[0,0,10,10],atol=1e-8)

    def test_kmeans_score_direction_and_equivalent_margin(self):
        X = np.array([[0.0], [0.1], [4.0], [4.1]])
        y = np.array([0, 0, 1, 1])
        relative = ClassConditionalKMeans(1, score_mode='relative_distance').fit(X, y)
        margin = ClassConditionalKMeans(1, score_mode='normalized_margin').fit(X, y)
        for model in (relative, margin):
            score = model.predict_proba(np.array([[0.05], [4.05]]))[:, 1]
            self.assertLess(score[0], score[1])
        np.testing.assert_allclose(
            relative.predict_proba(X)[:, 1], margin.predict_proba(X)[:, 1]
        )

    def test_kmeans_standardizer_uses_training_rows_only(self):
        candidate = next(
            item for item in kmeans_improvement_candidates(42)
            if item['k'] == 2 and item['scaling'] == 'standard'
        )
        X_train = np.array([[0.0, 10.0], [2.0, 14.0]])
        X_dev = np.array([[100.0, 1000.0]])
        candidate['model'].fit(X_train, [0, 1])
        scaler = candidate['model'].named_steps['standardscaler']
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))
        self.assertFalse(np.allclose(scaler.mean_, X_dev.mean(axis=0)))

    def test_kmeans_improvement_selection_uses_auroc_then_ap_deterministically(self):
        def record(config, auroc, ap):
            return {'config': config, 'development_metrics': {
                'auroc': auroc, 'average_precision': ap,
            }}
        records = [record('k=2', .60, .10), record('k=3', .61, .09),
                   record('k=4', .61, .12)]
        self.assertEqual(select_kmeans_improvement_candidate(records)['config'], 'k=4')

    def test_kmeans_controlled_selection_is_development_only_and_cap_safe(self):
        X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = ClassConditionalKMeans(1).fit(X, y)
        scores = model.predict_proba(X)[:, 1]
        record = {
            'family': 'kmeans', 'config': 'k=4', 'feature_set': 'original_7',
            'model': model, 'X': pd.DataFrame(X, columns=['x']),
        }
        selected = select_kmeans_controlled(record, y, scores)
        self.assertEqual(selected['original_fixed_0_5']['threshold'], .5)
        for key in ('recall_first_fpr_0_10', 'recall_first_fpr_0_20'):
            self.assertLessEqual(
                selected[key]['development_metrics']['false_positive_rate'],
                float(key.rsplit('_', 2)[-2] + '.' + key.rsplit('_', 1)[-1]),
            )
        changed_scores = np.full_like(scores, .1)
        changed = select_kmeans_controlled(record, y, changed_scores)
        self.assertNotEqual(
            selected['development_f1']['threshold'],
            changed['development_f1']['threshold'],
        )

    def test_threshold_is_selected_from_development_scores(self):
        threshold, method = select_classification_threshold(
            np.array([0, 0, 0, 1, 1, 1]), np.array([.05, .10, .20, .30, .70, .80])
        )
        self.assertEqual(method, 'development_f1')
        self.assertGreater(threshold, .2)
        metrics = classification_metrics(
            np.array([0, 0, 0, 1, 1, 1]), np.array([.05, .10, .20, .30, .70, .80]), threshold
        )
        self.assertIn('threshold_0_5', metrics)
        self.assertEqual(metrics['threshold_0_5']['threshold'], .5)

    def test_threshold_falls_back_with_insufficient_class_support(self):
        threshold, method = select_classification_threshold(np.array([0, 0, 0]), np.array([.1, .2, .3]))
        self.assertEqual((threshold, method), (.5, 'fixed_0.5_insufficient_class_support'))

    def test_joint_candidate_selection_uses_development_f1_then_average_precision(self):
        def record(name, f1, ap, threshold):
            return {
                'family': name, 'config': 'same', 'feature_set': 'original_7',
                'threshold': threshold,
                'development_metrics': {'f1': f1, 'average_precision': ap,
                                        'auroc': .5, 'threshold': threshold},
            }

        records = [record('lower-f1', .2, .99, .2),
                   record('higher-f1', .4, .10, .2),
                   record('higher-ap', .4, .20, .2)]
        self.assertEqual(select_classification_candidate(records)['family'], 'higher-ap')

    def test_candidate_ties_are_deterministic(self):
        records = [
            {'family': 'a', 'config': 'same', 'feature_set': 'original_7',
             'threshold': .4,
             'development_metrics': {'f1': .5, 'average_precision': .2, 'auroc': .5, 'threshold': .4}},
            {'family': 'b', 'config': 'same', 'feature_set': 'original_7',
             'threshold': .4,
             'development_metrics': {'f1': .5, 'average_precision': .2, 'auroc': .5, 'threshold': .4}},
        ]
        self.assertEqual(select_classification_candidate(records)['family'], 'b')

    def test_classification_metrics_report_zero_positive_predictions(self):
        metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([.1, .2, .3, .4]), .5)
        self.assertEqual(metrics['threshold'], .5)
        self.assertEqual(metrics['false_positives'], 0)
        self.assertEqual(metrics['false_negatives'], 2)
        self.assertEqual(metrics['true_negatives'], 2)
        self.assertEqual(metrics['true_positives'], 0)
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['false_positive_rate'], 0.0)

    def test_recall_first_respects_cap_and_maximizes_development_recall(self):
        def record(name, recall, fp, precision, ap, fpr):
            return {
                'family': name, 'config': 'config', 'feature_set': 'original_7',
                'threshold': .5, 'development_metrics': {
                    'recall': recall, 'false_positives': fp, 'precision': precision,
                    'average_precision': ap, 'false_positive_rate': fpr,
                    'f1': .1, 'auroc': .5,
                },
            }

        records = [
            record('too-many-false-alarms', 1.0, 30, .1, .9, .30),
            record('eligible-high-recall', .8, 8, .2, .2, .08),
            record('eligible-low-recall', .5, 0, 1.0, .9, 0.0),
        ]
        selected = select_recall_first_candidate(records, .10)
        self.assertEqual(selected['family'], 'eligible-high-recall')
        self.assertIsNone(select_recall_first_candidate([records[0]], .01))

    def test_recall_first_tie_breaks_are_deterministic(self):
        def record(name, fp, precision, ap, threshold):
            return {
                'family': name, 'config': 'config', 'feature_set': 'original_7',
                'threshold': threshold, 'development_metrics': {
                    'recall': .8, 'false_positives': fp, 'precision': precision,
                    'average_precision': ap, 'false_positive_rate': fp / 100,
                    'f1': .1, 'auroc': .5,
                },
            }

        records = [
            record('more-fp', 5, .9, .9, .8),
            record('best-precision', 2, .8, .1, .6),
            record('best-average-precision', 2, .8, .2, .5),
        ]
        self.assertEqual(select_recall_first_candidate(records, .10)['family'],
                         'best-average-precision')


if __name__ == '__main__':
    unittest.main()

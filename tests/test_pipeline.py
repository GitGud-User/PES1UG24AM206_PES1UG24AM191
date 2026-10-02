import unittest
import numpy as np
import pandas as pd
from src.data import assign_labels, past_counts, prepare, FEATURES
from src.experiment import split_data
from src.models import ClassConditionalKMeans, JointKMeansRegressor


class PipelineTests(unittest.TestCase):
    def test_counts_exact_boundaries_and_ties(self):
        t = pd.to_datetime(['2000-01-01','2000-01-02','2000-01-02','2000-01-03 00:00:01'], format='mixed')
        np.testing.assert_array_equal(past_counts(t,1), [0,1,1,0])
        # Appending future events cannot alter past features.
        np.testing.assert_array_equal(past_counts(t,7)[:3], past_counts(t[:3],7))

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

    def test_centroid_models_use_features_at_inference(self):
        X = np.array([[0,0],[0,.1],[4,4],[4,4.1]])
        c = ClassConditionalKMeans(1).fit(X,[0,0,1,1])
        np.testing.assert_array_equal(c.predict(X),[0,0,1,1])
        r = JointKMeansRegressor(2).fit(X,[0,0,10,10])
        np.testing.assert_allclose(r.predict(X),[0,0,10,10],atol=1e-8)


if __name__ == '__main__':
    unittest.main()

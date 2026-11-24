import ast
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import f_classif

from test_data_quality import CLASSES, FEATURES, notebook_cell, notebook_data


class PreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = notebook_data()

    def test_split_sizes_and_no_overlap(self):
        data = self.data
        splits = [data['X_train'], data['X_val'], data['X_test']]
        self.assertEqual([len(x) for x in splits], [9480, 2031, 2032])
        ids = [set(x.index) for x in splits]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(ids[i] & ids[j])
                rows_i = set(map(tuple, splits[i].to_numpy()))
                rows_j = set(map(tuple, splits[j].to_numpy()))
                self.assertFalse(rows_i & rows_j)
        self.assertEqual(set.union(*ids), set(data['X'].index))

    def test_labels_and_features_stay_aligned(self):
        for split in ['train', 'val', 'test']:
            with self.subTest(split=split):
                x = self.data['X_' + split]
                y = self.data['y_' + split]
                self.assertEqual(list(x.columns), FEATURES)
                pd.testing.assert_index_equal(x.index, y.index)
                pd.testing.assert_frame_equal(x, self.data['X'].loc[x.index])
                pd.testing.assert_series_equal(y, self.data['y'].loc[x.index])

    def test_class_proportions(self):
        overall = self.data['y'].value_counts(normalize=True).sort_index()
        for split in ['train', 'val', 'test']:
            with self.subTest(split=split):
                y = self.data['y_' + split]
                self.assertEqual(set(y), CLASSES)
                actual = y.value_counts(normalize=True).sort_index()
                np.testing.assert_allclose(actual, overall, atol=0.001, rtol=0)

    def test_notebook_training_calls_use_training_data(self):
        # Check both tuning and main experiments without rerunning the 63 fits.
        for name, label in [('tuning_rows', 'y_train'), ('experiment_rows', 'y_noisy')]:
            tree = ast.parse(notebook_cell(name))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name)
                     and node.func.id == 'fit_without_warning']
            with self.subTest(cell=name):
                self.assertEqual(len(calls), 1)
                self.assertEqual(ast.unparse(calls[0].args[1]), 'X_train')
                self.assertEqual(ast.unparse(calls[0].args[2]), label)

    def test_preprocessing_uses_training_statistics(self):
        data = self.data
        x, y = data['X_train'], data['y_train']
        configs = {
            'LR': {'C': 1},
            'RF': {'depth': 12, 'leaf': 1},
            'MLP': {'alpha': 0.01, 'lr': 0.001}
        }
        for name, config in configs.items():
            for k in [10, 16]:
                with self.subTest(model=name, features=k):
                    pipeline = data['build_model'](name, dict(config, k=k))
                    # Keep the real preprocessing; skip expensive model training.
                    pipeline.set_params(model=DummyClassifier(strategy='most_frequent'))
                    data['fit_without_warning'](pipeline, x, y)
                    if name == 'RF':
                        self.assertNotIn('scaler', pipeline.named_steps)
                        selector_input = x
                    else:
                        scaler = pipeline.named_steps['scaler']
                        self.assertEqual(scaler.n_samples_seen_, len(x))
                        np.testing.assert_allclose(scaler.mean_, x.mean().to_numpy())
                        np.testing.assert_allclose(scaler.var_, x.var(ddof=0).to_numpy())
                        selector_input = scaler.transform(x)
                        mean_before = scaler.mean_.copy()
                        var_before = scaler.var_.copy()
                    if k == 10:
                        selector = pipeline.named_steps['feature_selection']
                        scores, _ = f_classif(selector_input, y)
                        np.testing.assert_allclose(selector.scores_, scores)
                        self.assertEqual(int(selector.get_support().sum()), 10)
                        support_before = selector.get_support().copy()
                    else:
                        self.assertNotIn('feature_selection', pipeline.named_steps)

                    for split in ['val', 'test']:
                        held_out = data['X_' + split]
                        transformed = (pipeline[:-1].transform(held_out)
                                       if len(pipeline.steps) > 1 else held_out.to_numpy())
                        self.assertEqual(transformed.shape, (len(held_out), k))
                        self.assertTrue(np.isfinite(transformed).all())
                        self.assertEqual(len(pipeline.predict(held_out)), len(held_out))
                    if name != 'RF':
                        np.testing.assert_array_equal(scaler.mean_, mean_before)
                        np.testing.assert_array_equal(scaler.var_, var_before)
                        self.assertEqual(scaler.n_samples_seen_, len(x))
                    if k == 10:
                        np.testing.assert_array_equal(selector.get_support(), support_before)


if __name__ == '__main__':
    unittest.main()

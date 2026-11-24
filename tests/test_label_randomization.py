"""Run with: python -m unittest discover -s tests -p 'test_label_randomization.py' -v"""

import ast
import unittest

import numpy as np
import pandas as pd

from test_data_quality import CLASSES, NOTEBOOK, notebook_cell


class LabelRandomizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the actual function without running experiments or downloading data.
        tree = ast.parse(notebook_cell('randomize_train_labels'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == 'randomize_train_labels')
        module = ast.Module(body=[function], type_ignores=[])
        namespace = {'np': np}
        exec(compile(module, str(NOTEBOOK), 'exec'), namespace)
        cls.randomize = staticmethod(namespace['randomize_train_labels'])
        cls.classes = np.array(sorted(CLASSES))
        cls.labels = np.tile(cls.classes, 1000)

    def test_zero_noise_keeps_labels(self):
        for seed in [17, 29, 43]:
            with self.subTest(seed=seed):
                result = self.randomize(self.labels, 0, seed, self.classes)
                np.testing.assert_array_equal(result, self.labels)

    def test_same_seed_repeats_result(self):
        for p in [0.25, 0.5, 0.75, 1.0]:
            for seed in [17, 29, 43]:
                with self.subTest(p=p, seed=seed):
                    first = self.randomize(self.labels, p, seed, self.classes)
                    self.randomize(self.labels, p, seed + 1, self.classes)
                    second = self.randomize(self.labels, p, seed, self.classes)
                    np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_labels(self):
        first = self.randomize(self.labels, 0.5, 17, self.classes)
        second = self.randomize(self.labels, 0.5, 29, self.classes)
        self.assertFalse(np.array_equal(first, second))

    def test_input_is_not_modified(self):
        for use_series in [False, True]:
            with self.subTest(series=use_series):
                original = (pd.Series(self.labels.copy(), name='Class')
                            if use_series else self.labels.copy())
                before = original.copy()
                result = self.randomize(original, 1, 17, self.classes)
                np.testing.assert_array_equal(original, before)
                self.assertFalse(np.shares_memory(result, np.asarray(original)))
                result[0] = 'CHANGED'
                np.testing.assert_array_equal(original, before)

    def test_output_shape_and_classes(self):
        for p in [0, 0.25, 0.5, 0.75, 1]:
            with self.subTest(p=p):
                result = self.randomize(self.labels, p, 17, self.classes)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, self.labels.shape)
                self.assertTrue(np.isin(result, self.classes).all())

    def test_changed_fraction_matches_redraw_rule(self):
        labels = np.tile(self.classes, 10000)
        for p in [0.25, 0.5, 0.75, 1.0]:
            for seed in [17, 29, 43]:
                with self.subTest(p=p, seed=seed):
                    result = self.randomize(labels, p, seed, self.classes)
                    # A redraw can pick the original class: expected change is p * 6/7.
                    expected = p * (len(self.classes) - 1) / len(self.classes)
                    actual = np.mean(result != labels)
                    self.assertAlmostEqual(actual, expected, delta=0.01)

    def test_full_noise_draws_all_classes(self):
        # Start with one class to check the replacement distribution independently.
        labels = np.repeat(self.classes[0], 70000)
        result = self.randomize(labels, 1, 17, self.classes)
        for name in self.classes:
            with self.subTest(bean_class=name):
                self.assertAlmostEqual(np.mean(result == name), 1 / 7, delta=0.01)


if __name__ == '__main__':
    unittest.main()

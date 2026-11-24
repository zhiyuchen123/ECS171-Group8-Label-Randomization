"""Check the saved CSVs without training models or writing any results.

Run: python -m unittest discover -s tests -p 'test_experiment_results.py' -v
Adapted from the earlier reports/experiments/check_noise_results.py audit.
"""

from collections import Counter
import csv
from itertools import product
import math
from pathlib import Path
from statistics import mean, stdev
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('LR', 'RF', 'MLP')
SEEDS = (17, 29, 43)
LEVELS = (0, 0.25, 0.5, 0.75, 1)
METRICS = (
    'actual_changed_fraction', 'train_accuracy_on_noisy_labels',
    'train_accuracy_on_clean_labels', 'validation_macro_f1',
    'test_macro_f1', 'test_accuracy', 'test_balanced_accuracy'
)
SUMMARY_FIELDS = {
    'test_macro_f1_mean': 'test_macro_f1',
    'test_accuracy_mean': 'test_accuracy',
    'test_balanced_accuracy_mean': 'test_balanced_accuracy',
    'train_noisy_mean': 'train_accuracy_on_noisy_labels',
    'train_clean_mean': 'train_accuracy_on_clean_labels'
}


def read_rows(filename):
    with (ROOT / filename).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


class ExperimentResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = read_rows('all_experiments.csv')
        cls.summary = read_rows('summary_by_noise.csv')

    def test_all_experiment_combinations(self):
        actual = Counter((r['model'], float(r['seed']), float(r['noise_p']))
                         for r in self.runs)
        expected = Counter(product(MODELS, SEEDS, LEVELS))
        self.assertEqual(actual, expected)

    def test_all_summary_combinations(self):
        actual = Counter((r['model'], float(r['noise_p'])) for r in self.summary)
        self.assertEqual(actual, Counter(product(MODELS, LEVELS)))

    def test_metric_ranges_and_runtime(self):
        for row in self.runs:
            with self.subTest(model=row['model'], seed=row['seed'], p=row['noise_p']):
                for field in METRICS:
                    value = float(row[field])
                    self.assertTrue(math.isfinite(value), field)
                    self.assertGreaterEqual(value, 0, field)
                    self.assertLessEqual(value, 1, field)
                runtime = float(row['runtime_seconds'])
                self.assertTrue(math.isfinite(runtime))
                self.assertGreaterEqual(runtime, 0)

    def test_recorded_noise_fraction_matches_across_models(self):
        # Equal fractions are necessary, but do not prove the full label arrays match.
        for seed, p in product(SEEDS, LEVELS):
            with self.subTest(seed=seed, p=p):
                group = [r for r in self.runs
                         if float(r['seed']) == seed and float(r['noise_p']) == p]
                self.assertEqual(len(group), 3)
                values = [float(r['actual_changed_fraction']) for r in group]
                self.assertTrue(all(math.isclose(v, values[0], rel_tol=0, abs_tol=1e-12)
                                    for v in values))

    def test_zero_noise_training_metrics(self):
        clean = [r for r in self.runs if float(r['noise_p']) == 0]
        self.assertEqual(len(clean), 9)
        for row in clean:
            with self.subTest(model=row['model'], seed=row['seed']):
                self.assertEqual(float(row['actual_changed_fraction']), 0)
                self.assertAlmostEqual(float(row['train_accuracy_on_noisy_labels']),
                                       float(row['train_accuracy_on_clean_labels']),
                                       delta=1e-12)

    def test_summary_means(self):
        for row in self.summary:
            group = [r for r in self.runs if r['model'] == row['model']
                     and float(r['noise_p']) == float(row['noise_p'])]
            self.assertEqual(len(group), 3)
            for output, source in SUMMARY_FIELDS.items():
                with self.subTest(model=row['model'], p=row['noise_p'], field=output):
                    expected = mean(float(r[source]) for r in group)
                    self.assertAlmostEqual(float(row[output]), expected, delta=1e-12)

    def test_summary_standard_deviation(self):
        for row in self.summary:
            with self.subTest(model=row['model'], p=row['noise_p']):
                values = [float(r['test_macro_f1']) for r in self.runs
                          if r['model'] == row['model']
                          and float(r['noise_p']) == float(row['noise_p'])]
                self.assertEqual(len(values), 3)
                # pandas groupby.std() uses the sample SD (ddof=1).
                self.assertAlmostEqual(float(row['test_macro_f1_sd']),
                                       stdev(values), delta=1e-12)


if __name__ == '__main__':
    unittest.main()

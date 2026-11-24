import ast
from contextlib import redirect_stdout
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'ECS171_Group8_Project.ipynb'
FEATURES = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1',
    'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]
CLASSES = {'BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA'}


def notebook_cell(name):
    """Find a cell by a variable or function it defines, not its position."""
    cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
    matches = []
    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                matches.append(source)
                break
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                matches.append(source)
                break
    if len(matches) != 1:
        raise ValueError(f'Expected one notebook cell defining {name}, found {len(matches)}')
    return matches[0]


@lru_cache(maxsize=1)
def notebook_data():
    # Run the original cells in a temporary folder so outputs stay untouched.
    cache = Path(os.environ.get(
        'DRY_BEAN_ZIP', ROOT / 'data/raw/dry_bean_dataset.zip'
    )).resolve()
    namespace = {'__name__': '__test_notebook__'}
    previous_dir = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='bean-tests-') as folder:
        raw_dir = Path(folder) / 'data/raw'
        raw_dir.mkdir(parents=True)
        if cache.is_file():
            shutil.copyfile(cache, raw_dir / 'dry_bean_dataset.zip')
        elif 'DRY_BEAN_ZIP' in os.environ:
            raise FileNotFoundError(cache)
        try:
            os.chdir(folder)
            with redirect_stdout(io.StringIO()):
                for name in ['SEEDS', 'URL', 'audit', 'X', 'build_model']:
                    exec(compile(notebook_cell(name), str(NOTEBOOK), 'exec'), namespace)
        finally:
            os.chdir(previous_dir)
    return namespace


class DataQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = notebook_data()

    def test_dataset_size_and_duplicate_removal(self):
        raw = self.data['df']
        clean = self.data['clean_df']
        self.assertEqual(raw.shape, (13611, 17))
        self.assertEqual(int(raw.duplicated().sum()), 68)
        self.assertEqual(clean.shape, (13543, 17))
        pd.testing.assert_frame_equal(
            clean, raw.drop_duplicates().reset_index(drop=True)
        )

    def test_no_missing_or_duplicate_rows(self):
        clean = self.data['clean_df']
        self.assertFalse(clean.isna().any().any())
        self.assertFalse(clean.duplicated().any())

    def test_numeric_features_and_column_order(self):
        self.assertEqual(self.data['FEATURES'], FEATURES)
        features = self.data['clean_df'][FEATURES]
        self.assertTrue(all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes))
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertNotIn('Class', self.data['X'].columns)

    def test_all_seven_classes_remain(self):
        self.assertEqual(set(self.data['df']['Class']), CLASSES)
        self.assertEqual(set(self.data['clean_df']['Class']), CLASSES)


if __name__ == '__main__':
    unittest.main()

# Data and preprocessing tests

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

No extra test package is needed. The tests use Python's `unittest` module.

`test_data_quality.py` checks the original row count, duplicate removal, missing
values, numeric features, column order, and the seven bean classes.

`test_preprocessing.py` checks the 70/15/15 split, overlapping rows, label
alignment, class proportions, and preprocessing for all three models with
10 or 16 features. It checks that scaler statistics and feature scores come
from the training split and stay unchanged when processing validation and test
data. It also checks the training arguments in the notebook's two experiment loops.

The tests run the notebook's own setup, loading, cleaning, split, and model
definition cells in a temporary folder. They do not copy the preprocessing into
a separate implementation. Cells are found by the variables or functions they
define, so moving a cell does not change which code gets tested.

The dataset is downloaded from UCI unless `data/raw/dry_bean_dataset.zip` already
exists. For an offline run, point to a local copy of the original UCI ZIP:

```bash
DRY_BEAN_ZIP=/path/to/dry_bean_dataset.zip python -m unittest discover -s tests -v
```

A missing ZIP or failed download is an error, not a skipped test.

These are data and preprocessing checks, not a repeat of the model experiments.
The pipeline tests replace only the final classifier with `DummyClassifier` to
keep them quick. They do not verify the saved model or reproduce reported scores.
The existing model, CSV files, plots, and notebook outputs are not overwritten.

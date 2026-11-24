# ECS 171 Group 8: Memorization vs. Generalization

**UC Davis | ECS 171 – Machine Learning | Group 8**

## Project Overview

This project investigates how training-label randomization affects memorization and generalization in machine learning.

Inspired by Zhang et al. (2017), *Understanding Deep Learning Requires Rethinking Generalization*, we compare three classification models on the UCI Dry Bean dataset:

- Logistic Regression (LR)
- Random Forest (RF)
- Multilayer Perceptron (MLP)

We evaluate whether models can fit randomized training labels while maintaining good performance on unseen data.

## Dataset

We use the [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), which contains 16 numerical features and seven bean classes.

After removing 68 duplicate observations, 13,543 samples remain.

The dataset is divided using a stratified 70/15/15 split:

| Dataset | Samples |
|---|---:|
| Training | 9,480 |
| Validation | 2,031 |
| Testing | 2,032 |

## Methodology

Our experimental pipeline includes:

1. Data cleaning and exploratory data analysis.
2. Feature scaling and feature selection (10 vs. 16 features).
3. Training LR, RF, and MLP classifiers.
4. Hyperparameter tuning using validation macro-F1.
5. Randomizing training labels at five levels: 0%, 25%, 50%, 75%, and 100%.
6. Repeating experiments using three random seeds (17, 29, 43).
7. Comparing training accuracy and clean-test performance.

We evaluate 18 hyperparameter configurations and conduct 45 main experimental training runs.

## Key Results

Increasing training-label randomization generally reduces test performance.

At 100% label randomization, Random Forest achieves approximately 84.6% training accuracy on randomized labels, but only 0.122 test macro-F1.

This demonstrates that fitting training labels does not necessarily imply good generalization.

For clean-label prediction, MLP is selected as the final model, achieving approximately 93% test accuracy and 0.94 test macro-F1.

## Installation

Clone the repository:

```bash
git clone https://github.com/zhiyuchen123/ECS171-Group8-Label-Randomization.git
cd ECS171-Group8-Label-Randomization
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Reproducing the Experiments

From the repository root, open the notebook:

```bash
python -m notebook ECS171_Group8_Project.ipynb
```

Use a kernel from the Python environment where you installed the requirements. Restart the kernel and run all cells from top to bottom. Wait for the final model-saving cell to finish; opening the notebook alone only shows the saved outputs.

The notebook automatically downloads the UCI Dry Bean dataset and performs data preprocessing, exploratory data analysis, feature selection, model training, hyperparameter tuning, label-randomization experiments, and model evaluation.

Running the notebook generates experimental results, figures, and the trained model in the `outputs_notebook/` directory.

For convenience, selected results and visualizations are also included in the root directory of this GitHub repository.

Key output files include:

- `all_experiments.csv` – Results from 45 experimental runs.
- `summary_by_noise.csv` – Aggregated performance across noise levels.
- `hyperparameter_tuning.csv` – Hyperparameter tuning results.
- `macro_f1_vs_noise.png` – Generalization performance under label randomization.
- `best_clean_confusion_matrix.png` – Confusion matrix of the selected model.
- `best_model.joblib` – Saved final classification model.

### Using a new run in the demo

The app reads `best_model.joblib` and `summary_by_noise.csv` from the repository root, not from `outputs_notebook/`. To use a completed new run, stop the local app and copy the new files to the root. Copy the matching experiment records and figures too, so they stay together.

On macOS or Linux, run this from the repository root after the notebook finishes:

```bash
cp outputs_notebook/best_model.joblib \
   outputs_notebook/best_model_metadata.json \
   outputs_notebook/all_experiments.csv \
   outputs_notebook/summary_by_noise.csv \
   outputs_notebook/hyperparameter_tuning.csv \
   outputs_notebook/*.png .
```

This replaces the saved files in the root with the new run. Skip this step if you want to keep using the results already included in the repository. Run the tests below after copying, then restart the app so its cached model and results are reloaded. Copying files locally does not update the hosted demo; that requires updating its connected deployment repository.

## Tests

Run all tests from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

- [Data quality](tests/test_data_quality.py): duplicate removal, missing values, feature columns, and classes.
- [Preprocessing](tests/test_preprocessing.py): split overlap, label alignment, and training-only preprocessing.
- [Label randomization](tests/test_label_randomization.py): reproducibility, unchanged input labels, and the random redraw rule.
- [Experiment results](tests/test_experiment_results.py): complete experiment combinations, recorded noise fractions, metric ranges, and summary means and standard deviations.

The data and preprocessing tests need the UCI dataset; see [tests/README.md](tests/README.md) for the download and offline options. These tests do not rerun the full model experiments or verify the saved demo model's predictions.

## Web Demo

The web application demonstrates the prediction workflow:

**Input Features → Trained Model → Predicted Bean Class**

Users can enter the numerical features of a dry bean observation and obtain a predicted class.

To run the app locally, use the same Python environment as above and run this from the repository root:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal. The app uses the saved model and results, so you do not need to retrain before trying it. Press `Ctrl+C` in the terminal to stop it.

**Live Demo:** Our interactive Dry Bean Classification application is deployed using Streamlit.

[Try the Dry Bean Classifier](https://ecs171-group8-label-randomization-8es2wglcaqgwgqjgvjuxbj.streamlit.app/)

## Team Members

| Name | Contributions |
|------|---------------|
| Tianying Chen | Data cleaning, EDA, feature analysis, project coordination, and [tests](tests/) for data quality, preprocessing, label randomization, and experiment result consistency |
| Linjie Yang | Model implementation, hyperparameter tuning, experimental development, and web application development |
| Elsey Chen | Model evaluation, visualization, model comparison, result interpretation, GitHub repository management|

## References
1. Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. (2017). [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530). International Conference on Learning Representations.

2. UCI Machine Learning Repository. (2020). [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).

3. Koklu, M., and Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.

4. Frénay, B., and Verleysen, M. (2014). Classification in the presence of label noise: A survey. *IEEE Transactions on Neural Networks and Learning Systems*, 25(5), 845–869.

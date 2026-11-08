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
pip install -r requirements.txt
```

## Reproducing the Experiments

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `ECS171_Group8_Project.ipynb` and select **Kernel → Restart & Run All**.

The notebook automatically downloads the Dry Bean dataset and performs data preprocessing, EDA, feature selection, model training, hyperparameter tuning, label-randomization experiments, and evaluation.

Generated results, figures, and the selected model are saved in `outputs_notebook/`.

Important outputs include:

- `all_experiments.csv` – Results from 45 experiments
- `summary_by_noise.csv` – Aggregated model comparisons
- `macro_f1_vs_noise.png` – Generalization performance
- `best_model.joblib` – Saved final model

## Web Demo

The web application demonstrates the prediction workflow:

**Input Features → Trained Model → Predicted Bean Class**

Users can enter the numerical features of a dry bean observation and obtain a predicted class.

**Live Demo:** Our interactive Dry Bean Classification application is deployed using Streamlit.

[Try the Dry Bean Classifier](https://ecs171-group8-label-randomization-8es2wglcaqgwgqjgvjuxbj.streamlit.app/)

## Team Members

| Name | Contributions |
|---|---|
| Tianying Chen | Data cleaning, EDA, feature analysis, project coordination |
| Linjie Yang | Model implementation, hyperparameter tuning, GitHub and web demo |
| Elsey Chen | Model evaluation, visualization, comparison, and interpretation |

## References
1. Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. (2017). [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530). International Conference on Learning Representations.

2. UCI Machine Learning Repository. (2020). [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).

3. Koklu, M., and Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.

4. Frénay, B., and Verleysen, M. (2014). Classification in the presence of label noise: A survey. *IEEE Transactions on Neural Networks and Learning Systems*, 25(5), 845–869.


from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="ECS 171 | Label Randomization Study",
    page_icon="🫘",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_model.joblib"
RESULTS_PATH = ROOT / "summary_by_noise.csv"


# ============================================================
# Feature Names
# ============================================================

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]


# ============================================================
# Example Bean from the Test Dataset
# Actual Class: SEKER
# ============================================================

EXAMPLE_BEAN = {
    "Area": 33319.0,
    "Perimeter": 668.6010000000001,
    "MajorAxisLength": 225.01723780818773,
    "MinorAxisLength": 188.8365102736357,
    "AspectRation": 1.1915981580157562,
    "Eccentricity": 0.5438088924235495,
    "ConvexArea": 33732.0,
    "EquivDiameter": 205.96861020803846,
    "Extent": 0.7960387996941896,
    "Solidity": 0.9877564330605952,
    "roundness": 0.9366293850989834,
    "Compactness": 0.915345918447426,
    "ShapeFactor1": 0.00675342110532092,
    "ShapeFactor2": 0.0029244526283231254,
    "ShapeFactor3": 0.837858150418362,
    "ShapeFactor4": 0.9983902287170083,
}


# ============================================================
# Load Saved Model and Experimental Results
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    results = pd.read_csv(RESULTS_PATH)

    required_columns = {
        "model",
        "noise_p",
        "test_macro_f1_mean",
        "test_macro_f1_sd",
        "train_noisy_mean",
        "train_clean_mean",
    }

    missing = required_columns.difference(results.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return results.sort_values(
        ["noise_p", "model"]
    )


# ============================================================
# Main Page
# ============================================================

st.title("🫘 Memorization vs. Generalization")

st.subheader("ECS 171 — Group 8")

st.write(
    "Training-Label Randomization on the UCI Dry Bean Dataset"
)

st.write(
    "This project investigates how machine learning models "
    "respond to randomized training labels. We compare "
    "Logistic Regression, Random Forest, and MLP, focusing "
    "on memorization and generalization."
)

st.divider()


# ============================================================
# Three Main Tabs
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📈 Experiment Results",
    "🧠 Memorization vs. Generalization",
    "🫘 Live Classification",
])


# ============================================================
# TAB 1: Experiment Results
# ============================================================

with tab1:

    st.header("Experiment Results")

    st.write(
        "How does increasing training-label randomization "
        "affect model performance on clean test data?"
    )

    try:
        results = load_results()

    except Exception as error:
        st.error(
            f"Unable to load experiment results: {error}"
        )

    else:

        st.subheader("Test Macro-F1 vs. Label Randomization")

        chart_data = results.pivot(
            index="noise_p",
            columns="model",
            values="test_macro_f1_mean",
        )

        chart_data = chart_data.reindex(
            columns=["LR", "RF", "MLP"]
        )

        chart_data.index.name = (
            "Label Randomization Probability"
        )

        st.line_chart(
            chart_data,
            height=420,
        )

        st.caption(
            "Each point represents mean clean-test Macro-F1 "
            "across three random seeds: 17, 29, and 43. "
            "Only training labels were randomized."
        )

        st.divider()

        st.subheader("Explore Results by Noise Level")

        noise_level = st.select_slider(
            "Select label randomization probability",
            options=sorted(
                results["noise_p"].unique().tolist()
            ),
            value=0.75,
            format_func=lambda x: f"p = {x:.2f}",
            key="results_noise",
        )

        selected_results = results[
            results["noise_p"] == noise_level
        ].copy()

        selected_results["Test Macro-F1"] = (
            selected_results["test_macro_f1_mean"]
            .map(lambda x: f"{x:.4f}")
        )

        selected_results["Standard Deviation"] = (
            selected_results["test_macro_f1_sd"]
            .map(lambda x: f"{x:.4f}")
        )

        display_table = selected_results[
            [
                "model",
                "Test Macro-F1",
                "Standard Deviation",
            ]
        ].rename(
            columns={"model": "Model"}
        )

        st.dataframe(
            display_table,
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "Key finding: As training labels become less "
            "informative, clean-test performance eventually "
            "decreases sharply. However, different models "
            "respond differently to label noise."
        )


# ============================================================
# TAB 2: Memorization vs. Generalization
# ============================================================

with tab2:

    st.header("Memorization vs. Generalization")

    st.write(
        "Can a model fit randomized training labels "
        "without learning patterns that generalize?"
    )

    try:
        results = load_results()

    except Exception as error:
        st.error(
            f"Unable to load experiment results: {error}"
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            selected_model = st.selectbox(
                "Select Model",
                ["RF", "LR", "MLP"],
            )

        with col2:

            selected_noise = st.select_slider(
                "Select Label Randomization Level",
                options=sorted(
                    results["noise_p"].unique().tolist()
                ),
                value=1.0,
                format_func=lambda x: f"p = {x:.2f}",
                key="memory_noise",
            )

        selected_row = results[
            (results["model"] == selected_model)
            & (results["noise_p"] == selected_noise)
        ].iloc[0]

        st.divider()

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                "Training Accuracy (Supplied Labels)",
                f"{selected_row['train_noisy_mean']:.2%}",
            )

        with metric2:

            st.metric(
                "Clean-Test Macro-F1",
                f"{selected_row['test_macro_f1_mean']:.4f}",
            )

        with metric3:

            st.metric(
                "Training Accuracy (Original Labels)",
                f"{selected_row['train_clean_mean']:.2%}",
            )

        st.divider()

        st.subheader("Training Fit vs. Test Performance")

        comparison = results[
            results["model"] == selected_model
        ].set_index("noise_p")

        comparison = comparison[
            [
                "train_noisy_mean",
                "train_clean_mean",
                "test_macro_f1_mean",
            ]
        ]

        comparison = comparison.rename(
            columns={
                "train_noisy_mean":
                    "Supplied-label Training Accuracy",
                "train_clean_mean":
                    "Original-label Training Accuracy",
                "test_macro_f1_mean":
                    "Clean-Test Macro-F1",
            }
        )

        st.line_chart(
            comparison,
            height=400,
        )

        st.caption(
            "Training accuracy and test Macro-F1 are different "
            "metrics. They are shown together to illustrate "
            "the difference between fitting training labels "
            "and generalizing to unseen observations."
        )

        if (
            selected_model == "RF"
            and selected_noise == 1.0
        ):

            st.success(
                "Key finding: Random Forest achieved 84.61% "
                "training accuracy against randomized labels, "
                "but its clean-test Macro-F1 was only 0.1222."
            )

            st.write(
                "This suggests that the model could fit many "
                "randomized training labels without learning "
                "useful relationships for unseen data."
            )

        st.info(
            "Important: p represents the probability of "
            "redrawing a training label. Because the original "
            "class can be drawn again, the expected fraction "
            "of labels actually changed is 6p/7. "
            "These results come from previously completed "
            "experiments, not live retraining."
        )


# ============================================================
# TAB 3: Live Classification
# ============================================================

with tab3:

    st.header("Live Dry Bean Classification")

    st.write(
        "This demonstration uses our selected MLP model, "
        "trained with clean labels, to classify a bean "
        "observation using 16 numerical features."
    )

    try:
        model = load_model()

    except Exception as error:
        st.error(
            f"Unable to load the saved model: {error}"
        )
        st.stop()

    # Initialize Feature Values
    for feature in FEATURES:

        if feature not in st.session_state:
            st.session_state[feature] = 0.0

    # Load Example Bean
    if st.button(
        "🫘 Load Example Bean",
        key="load_example",
    ):

        for feature in FEATURES:
            st.session_state[feature] = (
                EXAMPLE_BEAN[feature]
            )

        st.success(
            "Example bean loaded! Actual class: SEKER"
        )

    # Feature Input Form
    with st.expander(
        "View or Edit 16 Bean Features",
        expanded=False,
    ):

        with st.form("prediction_form"):

            left_col, right_col = st.columns(2)

            for i, feature in enumerate(FEATURES):

                with left_col if i < 8 else right_col:

                    st.number_input(
                        feature,
                        format="%.8f",
                        key=feature,
                    )

            submitted = st.form_submit_button(
                "Predict Bean Class",
                type="primary",
                use_container_width=True,
            )

    # Quick Prediction Button
    if st.button(
        "Predict with Current Values",
        type="primary",
        key="quick_predict",
    ):
        submitted = True

    # Prediction
    if submitted:

        values = {
            feature: st.session_state[feature]
            for feature in FEATURES
        }

        if all(
            value == 0.0
            for value in values.values()
        ):

            st.warning(
                "Please load an example bean or enter "
                "realistic feature values first."
            )

        else:

            input_data = pd.DataFrame(
                [values],
                columns=FEATURES,
            )

            try:

                prediction = model.predict(
                    input_data
                )[0]

                st.divider()

                st.success(
                    f"Predicted Bean Class: {prediction}"
                )

                if hasattr(model, "predict_proba"):

                    probabilities = (
                        model.predict_proba(input_data)[0]
                    )

                    probability_df = pd.DataFrame({
                        "Class": model.classes_,
                        "Probability": probabilities,
                    }).sort_values(
                        "Probability",
                        ascending=False,
                    )

                    confidence = (
                        probability_df.iloc[0]["Probability"]
                    )

                    st.metric(
                        "Top Predicted Probability",
                        f"{confidence:.2%}",
                    )

                    st.subheader(
                        "Prediction Probabilities"
                    )

                    st.bar_chart(
                        probability_df.set_index("Class")
                    )

                    with st.expander(
                        "View All Class Probabilities"
                    ):

                        st.dataframe(
                            probability_df,
                            hide_index=True,
                            use_container_width=True,
                        )

                    st.caption(
                        "The predicted probability is a model "
                        "estimate, not a guarantee that the "
                        "classification is correct."
                    )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "UC Davis ECS 171 | Group 8 | "
    "UCI Dry Bean Dataset | "
    "Educational Research Demonstration"
)

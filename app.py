from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ECS 171 Dry Bean Classifier",
    page_icon="🫘",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "best_model.joblib"

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


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🫘 Dry Bean Classification")
st.subheader("ECS 171 — Group 8")

st.write(
    "Memorization and Generalization "
    "under Training-Label Randomization"
)

st.markdown(
    """
    This interactive application demonstrates our final
    machine learning model trained on the UCI Dry Bean dataset.

    Enter the 16 numerical features of a dry bean observation
    to predict its class using our selected MLP classifier.
    """
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()

st.divider()
st.header("Enter Bean Features")

values = {}

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    for i, feature in enumerate(FEATURES):
        with col1 if i < 8 else col2:
            values[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.8f",
            )

    submitted = st.form_submit_button(
        "Predict Bean Class",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_data = pd.DataFrame([values], columns=FEATURES)

    try:
        prediction = model.predict(input_data)[0]

        st.divider()
        st.success(f"Predicted Bean Class: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            probability_df = pd.DataFrame(
                {
                    "Class": model.classes_,
                    "Probability": probabilities,
                }
            ).sort_values(
                "Probability", ascending=False
            )

            st.subheader("Prediction Probabilities")

            st.bar_chart(
                probability_df.set_index("Class")
            )

            st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True,
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "UC Davis ECS 171 | Group 8 | "
    "UCI Dry Bean Dataset | Educational Demonstration"
)

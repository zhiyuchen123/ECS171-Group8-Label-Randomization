from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ECS 171 Dry Bean Classifier",
    page_icon="🫘",
    layout="wide"
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
    "ShapeFactor4"
]

# Real example from our test dataset
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
    "ShapeFactor4": 0.9983902287170083
}

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🫘 Dry Bean Classification")
st.subheader("ECS 171 — Group 8")

st.write(
    "Memorization and Generalization under "
    "Training-Label Randomization"
)

st.write(
    "This interactive application demonstrates our final "
    "machine learning model trained on the UCI Dry Bean dataset."
)

st.write(
    "Enter the 16 numerical features of a dry bean observation "
    "to predict its class using our selected MLP classifier."
)

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

st.divider()

st.header("Enter Bean Features")

# Initialize input values
for feature in FEATURES:
    if feature not in st.session_state:
        st.session_state[feature] = 0.0

# Load a real example
if st.button("🫘 Load Example Bean"):
    for feature in FEATURES:
        st.session_state[feature] = EXAMPLE_BEAN[feature]
    st.success("Example bean loaded! Actual class: SEKER")

# Prediction form
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(FEATURES):
        with col1 if i < 8 else col2:
            st.number_input(
                feature,
                format="%.8f",
                key=feature
            )

    submitted = st.form_submit_button(
        "Predict Bean Class",
        type="primary",
        use_container_width=True
    )

if submitted:

    values = {
        feature: st.session_state[feature]
        for feature in FEATURES
    }

    input_data = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    try:
        prediction = model.predict(input_data)[0]

        st.divider()

        st.success(f"Predicted Bean Class: {prediction}")

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            probability_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            }).sort_values(
                "Probability",
                ascending=False
            )

            confidence = probability_df.iloc[0]["Probability"]

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

            st.subheader("Prediction Probabilities")

            st.bar_chart(
                probability_df.set_index("Class")
            )

            st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True
            )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "UC Davis ECS 171 | Group 8 | "
    "UCI Dry Bean Dataset | Educational Demonstration"
)

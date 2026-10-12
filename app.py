
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("Stroke Prediction Using Machine Learning")
st.write("Welcome to our capstone project.")
st.caption(
    "Developed as an academic project to demonstrate the application "
    "of machine learning in stroke risk prediction."
)

MODEL_PATH = Path(__file__).parent / "stroke_model.joblib"


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Make sure stroke_model.joblib "
        "is uploaded to the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error("The trained model could not be loaded.")
    st.exception(e)
    st.stop()

st.success("Trained model loaded successfully.")

st.subheader("Enter Patient Information")

with st.form("stroke_prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=120.0,
        value=45.0
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    heart_disease = st.selectbox(
        "Heart disease",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    ever_married = st.selectbox(
        "Ever married?",
        ["No", "Yes"]
    )

    work_type = st.selectbox(
        "Work type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )

    residence_type = st.selectbox(
        "Residence type",
        ["Urban", "Rural"]
    )

    avg_glucose_level = st.number_input(
        "Average glucose level",
        min_value=0.0,
        value=100.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=25.0
    )

    smoking_status = st.selectbox(
        "Smoking status",
        [
            "never smoked",
            "formerly smoked",
            "smokes",
            "Unknown"
        ]
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.warning(
                "The model predicts the positive stroke class. "
                "This is not a diagnosis. Seek assessment from a "
                "qualified healthcare professional."
            )
        else:
            st.info(
                "The model predicts the negative stroke class. "
                "This does not rule out stroke risk or guarantee safety."
            )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

            st.write("Model output probabilities:")

            st.json({
                str(label): round(float(probability), 4)
                for label, probability in zip(
                    model.classes_,
                    probabilities
                )
            })

    except Exception as e:
        st.error(
            "Prediction failed. The model may require different "
            "input categories or preprocessing."
        )
        st.exception(e)

st.divider()

st.caption(
    "Developed as an academic project to demonstrate the application "
    "of machine learning in stroke risk prediction."
)

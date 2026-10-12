import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(
page_title=“Stroke Prediction”,
page_icon=“🩺”,
layout=“centered”
)

st.title(“🩺 Stroke Prediction Using Machine Learning”)
st.write(“Welcome to our capstone project.”)
st.caption(“An academic machine learning project.”)

MODEL_PATH = Path(file).parent / “stroke_model.joblib”

@st.cache_resource
def load_model(path):
return joblib.load(path)

if not MODEL_PATH.exists():
st.error(
“The trained model file was not found. “
“Please check that stroke_model.joblib is in the app folder.”
)
st.stop()

try:
model = load_model(str(MODEL_PATH))
except Exception as e:
st.error(“The model could not be loaded.”)
st.code(str(e))
st.stop()

st.success(“Trained model loaded successfully.”)

st.subheader(“Model Information”)
st.write(“Model artifact: stroke_model.joblib”)
st.write(“Model type:”, type(model).name)

st.info(
“The model is loaded successfully. “
“Prediction inputs will be added after confirming “
“the exact features and preprocessing used during training.”
)

st.caption(
“Educational capstone demonstration only. “
“This application is not a medical diagnostic tool.”
)

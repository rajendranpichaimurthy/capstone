# Streamlit app - engine maintenance prediction
# loads the best model that the pipeline committed to the repo (models/best_model.joblib)
# takes the 6 sensor readings as input, puts them in a dataframe and predicts

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
META_PATH = ROOT / "models" / "best_model_metadata.json"


@st.cache_resource
def load_model(model_version):
    # load the model and its metadata once per model version
    # (model_version changes when the pipeline commits a new model, so the new one gets loaded)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


model, meta = load_model(MODEL_PATH.stat().st_mtime)
FEATURES = meta["features"]
THRESHOLD = meta.get("decision_threshold", 0.5)

st.set_page_config(page_title="Engine Maintenance Predictor", page_icon="🔧", layout="centered")
st.title("Engine Maintenance Predictor")
st.write("Enter the engine sensor readings to check if the engine needs maintenance.")

# input fields - default values are the median readings from the dataset
st.subheader("Sensor readings")
col1, col2 = st.columns(2)
with col1:
    engine_rpm = st.number_input("Engine rpm", min_value=0.0, max_value=3000.0, value=746.0, step=10.0)
    lub_oil_pressure = st.number_input("Lub oil pressure (bar)", min_value=0.0, max_value=10.0, value=3.16, step=0.1)
    fuel_pressure = st.number_input("Fuel pressure (bar)", min_value=0.0, max_value=25.0, value=6.20, step=0.1)
with col2:
    coolant_pressure = st.number_input("Coolant pressure (bar)", min_value=0.0, max_value=10.0, value=2.17, step=0.1)
    lub_oil_temp = st.number_input("Lub oil temp (°C)", min_value=0.0, max_value=150.0, value=76.82, step=0.5)
    coolant_temp = st.number_input("Coolant temp (°C)", min_value=0.0, max_value=200.0, value=78.35, step=0.5)

# save the inputs into a dataframe (same column order the model was trained on)
input_df = pd.DataFrame([{
    "engine_rpm": engine_rpm,
    "lub_oil_pressure": lub_oil_pressure,
    "fuel_pressure": fuel_pressure,
    "coolant_pressure": coolant_pressure,
    "lub_oil_temp": lub_oil_temp,
    "coolant_temp": coolant_temp,
}])[FEATURES]

st.subheader("Input data")
st.dataframe(input_df, hide_index=True)

if st.button("Predict", type="primary"):
    prob = float(model.predict_proba(input_df)[:, 1][0])
    if prob >= THRESHOLD:
        st.error(f"Maintenance needed - probability of fault {prob:.0%}")
    else:
        st.success(f"Engine looks normal - probability of fault {prob:.0%}")
    st.caption(f"The engine is flagged when the probability is {THRESHOLD:.0%} or more.")

with st.expander("About the model"):
    st.write(f"Model: **{meta['best_model']}** (picked using cross validation ROC-AUC {meta['cv_roc_auc']:.2f})")
    tm = meta["test_metrics"]
    st.write(f"Test set: ROC-AUC {tm['roc_auc']:.2f}, recall {tm['recall']:.2f}, "
             f"precision {tm['precision']:.2f}, accuracy {tm['accuracy']:.2f}")

"""Green Smart Campus: Streamlit app that predicts Total_Energy_Saved_kWh
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODELS_DIR = Path(__file__).parent / "models"

st.set_page_config(page_title="Green Campus Energy Saved", page_icon="🌱", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the saved preprocessing objects, the model and the feature information."""
    loaded = {}
    for name in ("preprocessor", "selector", "model", "feature_info"):
        with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
            loaded[name] = pickle.load(f)
    return loaded


art = load_artifacts()
info = art["feature_info"]

# ------------------------------------------------------------------ header and sidebar
st.title("Green Smart Campus: Energy Saved Predictor")
st.write("Enter the sensor readings below and the model predicts the **total energy saved (kWh)**.")

with st.sidebar:
    st.header("About the model")
    st.write("Ridge regression trained on 2,138 campus records, inside a pipeline of scaling, one-hot encoding and feature selection.")
    m = info["metrics"]
    st.metric("R2 on the test set", f"{m['test_r2']:.3f}")
    st.metric("Mean absolute error (test)", f"{m['test_mae']:.2f} kWh")
    st.caption("Features used by the model: " + ", ".join(info["selected_raw_features"]))
    st.caption("The data looks simulated, so treat the predictions as a demonstration.")


# ------------------------------------------------------------------ input widgets
def solar_input(default_value):
    """Solar generation can be typed in directly or calculated from sunlight (irradiance)."""
    method = st.radio("How do you want to give the solar generation?",
                      ["Enter it directly", "Calculate it from sunlight (irradiance)"], horizontal=True)
    if method == "Enter it directly":
        lo, hi = info["ranges"]["Solar_Generation_kWh"]
        return st.number_input(f"Solar generation (kWh), training range {lo:.1f} to {hi:.1f}",
                               min_value=0.0, value=round(default_value, 1), step=1.0)
    b_lo, b_hi = info["irradiance_ranges"]["Beam_Irradiance_Wm2"]
    d_lo, d_hi = info["irradiance_ranges"]["Diffuse_Irradiance_Wm2"]
    beam = st.number_input(f"Beam irradiance (W/m2), training range {b_lo:.0f} to {b_hi:.0f}",
                           min_value=0.0, value=float(round((b_lo + b_hi) / 2)), step=10.0)
    diffuse = st.number_input(f"Diffuse irradiance (W/m2), training range {d_lo:.0f} to {d_hi:.0f}",
                              min_value=0.0, value=float(round((d_lo + d_hi) / 2)), step=10.0)
    value = info["solar_from_irradiance_factor"] * (beam + diffuse)
    st.caption(f"In the data, solar generation = 0.05 x (beam + diffuse) = **{value:.2f} kWh**")
    return value


st.subheader("Inputs")
inputs = dict(info["defaults"])          # every column starts at its typical training value
for col in info["selected_raw_features"]:
    if col == "Solar_Generation_kWh":
        inputs[col] = solar_input(info["defaults"][col])
    elif col in info["cat_cols"]:
        inputs[col] = st.selectbox(col, info["categories"][col])
    elif col == "Is_Peak_Hour":
        inputs[col] = 1 if st.selectbox("Peak hour?", ["No", "Yes"]) == "Yes" else 0
    else:
        lo, hi = info["ranges"][col]
        inputs[col] = st.number_input(f"{col}, training range {lo:.1f} to {hi:.1f}",
                                      value=float(round(info["defaults"][col], 2)))

# ------------------------------------------------------------------ prediction
if st.button("Predict energy saved", type="primary"):
    row = pd.DataFrame([inputs])[info["feature_cols"]]
    features = art["selector"].transform(art["preprocessor"].transform(row))
    prediction = float(art["model"].predict(features)[0])
    low, high = (prediction * k for k in info["range_multipliers"])

    st.metric("Predicted energy saved", f"{prediction:.2f} kWh")
    st.write(f"Typical range (about 90% of cases): **{low:.2f} to {high:.2f} kWh**")

    ratio_min, ratio_mean, ratio_max = info["saved_to_solar_ratio"]
    if "Solar_Generation_kWh" in info["selected_raw_features"]:
        st.info(f"In the training data the energy saved was always between {ratio_min:.0%} and {ratio_max:.0%} "
                f"of the solar generation (on average {ratio_mean:.0%}), so the prediction is mostly a share of the solar generation.")

    for col in info["selected_raw_features"]:
        if col in info["ranges"] and not (info["ranges"][col][0] <= inputs[col] <= info["ranges"][col][1]):
            st.warning(f"{col} is outside the range seen in training, so the prediction is less reliable.")

    with st.expander("Inputs sent to the model"):
        st.dataframe(row.T.astype(str).rename(columns={0: "value"}))

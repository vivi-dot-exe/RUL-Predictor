import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RUL Predictor", page_icon="⚙️", layout="wide")

try:
  model = joblib.load("turbofan_model.pkl")
  feature_cols = joblib.load("feature_cols.pkl")
except Exception as e:
  st.error(f"Error loading model artifacts: {e}")
  st.stop()

# Calibrated C-MAPSS FD001 sensor operating profiles
PRESETS = {
    "Healthy (High RUL)": {
        "setting_1": 0.0002,
        "setting_2": 0.0001,
        "setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": 641.90,
        "sensor_3": 1581.0,
        "sensor_4": 1400.0,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 554.50,
        "sensor_8": 2388.05,
        "sensor_9": 9050.0,
        "sensor_10": 1.30,
        "sensor_11": 47.15,
        "sensor_12": 522.40,
        "sensor_13": 2388.05,
        "sensor_14": 8128.0,
        "sensor_15": 8.39,
        "sensor_16": 0.03,
        "sensor_17": 391.0,
        "sensor_20": 39.00,
        "sensor_21": 23.40,
    },
    "Moderate Wear": {
        "setting_1": 0.0002,
        "setting_2": 0.0001,
        "setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": 642.70,
        "sensor_3": 1591.0,
        "sensor_4": 1410.0,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 553.10,
        "sensor_8": 2388.10,
        "sensor_9": 9065.0,
        "sensor_10": 1.30,
        "sensor_11": 47.60,
        "sensor_12": 521.10,
        "sensor_13": 2388.10,
        "sensor_14": 8145.0,
        "sensor_15": 8.45,
        "sensor_16": 0.03,
        "sensor_17": 393.0,
        "sensor_20": 38.75,
        "sensor_21": 23.25,
    },
    "Critical (Low RUL)": {
        "setting_1": 0.0002,
        "setting_2": 0.0001,
        "setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": 643.50,
        "sensor_3": 1600.0,
        "sensor_4": 1422.0,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 551.80,
        "sensor_8": 2388.18,
        "sensor_9": 9085.0,
        "sensor_10": 1.30,
        "sensor_11": 48.00,
        "sensor_12": 519.80,
        "sensor_13": 2388.18,
        "sensor_14": 8175.0,
        "sensor_15": 8.51,
        "sensor_16": 0.03,
        "sensor_17": 396.0,
        "sensor_20": 38.45,
        "sensor_21": 23.08,
    },
}

preset_options = [
    "Healthy (High RUL)",
    "Moderate Wear",
    "Critical (Low RUL)",
    "Custom (Manual)",
]

if "preset" not in st.session_state:
  st.session_state.preset = "Healthy (High RUL)"
if "sensor_values" not in st.session_state:
  st.session_state.sensor_values = {
      col: float(PRESETS["Healthy (High RUL)"].get(col, 0.0))
      for col in feature_cols
  }

for col in feature_cols:
  if f"input_grid_{col}" not in st.session_state:
    st.session_state[f"input_grid_{col}"] = float(
        st.session_state.sensor_values[col]
    )

st.sidebar.header("Control Panel")
selected_preset = st.sidebar.selectbox(
    "Engine Profile Preset",
    preset_options,
    index=preset_options.index(st.session_state.preset),
    key="engine_profile_preset",
)

if selected_preset != st.session_state.preset:
  st.session_state.preset = selected_preset
  if selected_preset in PRESETS:
    for col in feature_cols:
      val = float(PRESETS[selected_preset].get(col, 0.0))
      st.session_state.sensor_values[col] = val
      st.session_state[f"input_grid_{col}"] = val
  elif selected_preset == "Custom (Manual)":
    for col in feature_cols:
      st.session_state.sensor_values[col] = 0.0
      st.session_state[f"input_grid_{col}"] = 0.0
  st.rerun()

st.title("RUL Predictor")
st.caption(
    "Enterprise prognostics and health management platform for turbofan"
    " engine degradation forecasting."
)

st.divider()

st.subheader("Telemetry & Sensor Inputs")
st.markdown("Adjust individual sensor readings or operating parameters below:")

# Render sensors in a clean 4-column grid (no horizontal table scrolling)
cols_per_row = 4
feature_list = list(feature_cols)

for i in range(0, len(feature_list), cols_per_row):
  cols = st.columns(cols_per_row)
  for j in range(cols_per_row):
    if i + j < len(feature_list):
      col_name = feature_list[i + j]
      with cols[j]:
        val = st.number_input(
            f"{col_name}",
            format="%.4f",
            key=f"input_grid_{col_name}",
        )
        st.session_state.sensor_values[col_name] = float(val)

st.divider()

# Prediction section
col_btn, col_metrics = st.columns([1, 3])

with col_btn:
  st.markdown("<br>", unsafe_allow_html=True)
  predict_btn = st.button(
      "Run Prediction", type="primary", use_container_width=True
  )

input_df = pd.DataFrame([st.session_state.sensor_values])

if predict_btn:
  prediction = model.predict(input_df)[0]

  with col_metrics:
    r1, r2, r3 = st.columns(3)
    with r1:
      st.metric("Estimated RUL", f"{prediction:.1f} Cycles")
    with r2:
      if prediction > 120:
        status = "Nominal / Healthy"
      elif prediction > 40:
        status = "Moderate Wear"
      else:
        status = "Critical / Service Required"
      st.metric("Health Status", status)
    with r3:
      ratio = min(max(prediction / 250.0, 0.0), 1.0)
      st.metric("Remaining Life Ratio", f"{int(ratio * 100)}%")

  st.progress(float(ratio))


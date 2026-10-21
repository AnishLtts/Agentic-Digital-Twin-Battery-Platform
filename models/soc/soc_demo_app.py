# SOC Estimation Demo: Agentic Digital Twin Battery Platform (Model 1)
import joblib
import numpy as np
import pandas as pd
import streamlit as st

DATA_FILE = "demo/soc/soc_test_drives.csv"
MODEL_FILE = "models/soc/soc_model.joblib"
SMOOTH_WINDOW = 60

st.set_page_config(page_title="SOC Estimation Demo", page_icon="🔋", layout="wide")


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_FILE)
    return saved["model"], saved["features"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


def add_features(part):
    part = part.copy()
    for window in [30, 120]:
        for col in ["Voltage", "Current", "Temperature"]:
            part[f"{col}_avg_{window}s"] = part[col].rolling(window, min_periods=1).mean()
    return part


model, FEATURES = load_model()
df = load_data()

# ---------------- Header ----------------
st.title("🔋 Battery SOC Estimation: Live Demo")
st.caption("Model 1 of the Agentic Digital Twin Battery Platform | "
           "LG HG2 cell | Tested on drive cycles the model has never seen")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
test_files = sorted(df.loc[df["Source_File"].str.contains("Mixed8"), "Source_File"].unique())
selected_file = st.sidebar.selectbox("Choose an unseen drive cycle", test_files)
use_smoothing = st.sidebar.checkbox("BMS smoothing (60 s)", value=True)

# ---------------- Predict ----------------
part = add_features(df[df["Source_File"] == selected_file].reset_index(drop=True))
part["SOC_Predicted"] = np.clip(model.predict(part[FEATURES]), 0, 100)
if use_smoothing:
    part["SOC_Predicted"] = part["SOC_Predicted"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
part["Error"] = (part["SOC_Predicted"] - part["SOC_Percent"]).abs()

# ---------------- Summary metrics ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ambient temperature", f"{part['Ambient_Temp_C'].iloc[0]} °C")
c2.metric("Drive duration", f"{part['Time_s'].iloc[-1] / 60:.0f} min")
c3.metric("Average error (MAE)", f"{part['Error'].mean():.2f} %")
c4.metric("Max error", f"{part['Error'].max():.2f} %")

# ---------------- Main chart ----------------
st.subheader("True SOC vs Predicted SOC")
chart = part.set_index("Time_s")[["SOC_Percent", "SOC_Predicted"]]
chart.columns = ["True SOC (%)", "Predicted SOC (%)"]
st.line_chart(chart, color=["#2ca02c", "#9467bd"])

# ---------------- Replay: pick any second ----------------
st.subheader("⏱ Replay the drive: what the BMS sees at any second")
second = st.slider("Time (s)", 0, len(part) - 1, len(part) // 2)
row = part.iloc[second]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Voltage", f"{row['Voltage']:.3f} V")
m2.metric("Current", f"{row['Current']:.2f} A")
m3.metric("Temperature", f"{row['Temperature']:.1f} °C")
m4.metric("True SOC", f"{row['SOC_Percent']:.1f} %")
m5.metric("Predicted SOC", f"{row['SOC_Predicted']:.1f} %",
          delta=f"{row['SOC_Predicted'] - row['SOC_Percent']:+.2f} % error",
          delta_color="off")

left, right = st.columns(2)
left.caption("Voltage (V)")
left.line_chart(part.set_index("Time_s")["Voltage"])
right.caption("Current (A): negative = driving, positive = regenerative braking")
right.line_chart(part.set_index("Time_s")["Current"], color="#ff7f0e")

# ---------------- Try your own values ----------------
with st.expander("🧪 Try your own sensor values"):
    st.write("Enter steady readings. The model assumes they have been constant for the last 2 minutes.")
    i1, i2, i3 = st.columns(3)
    v = i1.number_input("Voltage (V)", 2.5, 4.2, 3.7, 0.01)
    i = i2.number_input("Current (A)", -20.0, 5.0, -2.0, 0.1)
    t = i3.number_input("Temperature (°C)", -20.0, 45.0, 25.0, 1.0)
    manual = pd.DataFrame([{
        "Voltage": v, "Current": i, "Temperature": t,
        "Voltage_avg_30s": v, "Current_avg_30s": i, "Temperature_avg_30s": t,
        "Voltage_avg_120s": v, "Current_avg_120s": i, "Temperature_avg_120s": t,
    }])[FEATURES]
    soc = float(np.clip(model.predict(manual)[0], 0, 100))
    st.metric("Estimated SOC", f"{soc:.1f} %")
    st.progress(soc / 100)
# Step 5: Smooth the SOC predictions to remove cold-weather spikes
import joblib
import numpy as np
import pandas as pd

DATA_FILE = "data/soc/soc_dataset_1Hz.csv"
MODEL_FILE = "models/soc/soc_model.joblib"
SMOOTH_WINDOW = 60  # seconds

saved = joblib.load(MODEL_FILE)
model, FEATURES = saved["model"], saved["features"]

df = pd.read_csv(DATA_FILE)

# Same features as in training
groups = df.groupby("Source_File")
for window in [30, 120]:
    for col in ["Voltage", "Current", "Temperature"]:
        df[f"{col}_avg_{window}s"] = groups[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

test_df = df[df["Source_File"].str.contains("Mixed8")].copy()
test_df["SOC_Raw"] = np.clip(model.predict(test_df[FEATURES]), 0, 100)

# Smooth each file separately (uses only past predictions, like a real BMS)
test_df["SOC_Smooth"] = test_df.groupby("Source_File")["SOC_Raw"].transform(
    lambda s: s.rolling(SMOOTH_WINDOW, min_periods=1).mean()
)

def report(name, column):
    err = (test_df[column] - test_df["SOC_Percent"]).abs()
    print(f"{name:10s} MAE={err.mean():.2f}%  Max error={err.max():.2f}%")
    return err

print("=" * 50)
raw_err = report("Raw", "SOC_Raw")
smooth_err = report("Smoothed", "SOC_Smooth")
print("=" * 50)

print("\nMAE per temperature (Raw vs Smoothed):")
summary = pd.DataFrame({
    "Raw": raw_err.groupby(test_df["Ambient_Temp_C"]).mean(),
    "Smoothed": smooth_err.groupby(test_df["Ambient_Temp_C"]).mean(),
}).round(2)
print(summary)
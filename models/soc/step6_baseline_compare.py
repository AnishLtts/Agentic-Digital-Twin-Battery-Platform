# Step 6: Compare the ML model with a realistic coulomb-counting BMS
import joblib
import numpy as np
import pandas as pd

DATA_FILE = "data/soc/soc_dataset_1Hz.csv"
MODEL_FILE = "models/soc/soc_model.joblib"
CAPACITY_AH = 3.0

saved = joblib.load(MODEL_FILE)
model, FEATURES = saved["model"], saved["features"]

df = pd.read_csv(DATA_FILE)
groups = df.groupby("Source_File")
for window in [30, 120]:
    for col in ["Voltage", "Current", "Temperature"]:
        df[f"{col}_avg_{window}s"] = groups[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

test_df = df[df["Source_File"].str.contains("Mixed8")].copy()

# ML model (smoothed, as in Step 5)
test_df["SOC_ML"] = np.clip(model.predict(test_df[FEATURES]), 0, 100)
test_df["SOC_ML"] = test_df.groupby("Source_File")["SOC_ML"].transform(
    lambda s: s.rolling(60, min_periods=1).mean()
)


def coulomb_count(part, start_soc, current_offset_a):
    """Real-car coulomb counting: unknown start SOC + small sensor offset."""
    dt = part["Time_s"].diff().fillna(0)
    measured_current = part["Current"] + current_offset_a
    charge_ah = (measured_current * dt).cumsum() / 3600
    return np.clip(start_soc + 100 * charge_ah / CAPACITY_AH, 0, 100)


scenarios = {
    "CC perfect (lab conditions)": (100, 0.00),
    "CC sensor offset 0.05 A":     (100, -0.05),
    "CC wrong start SOC (90%)":    (90, 0.00),
    "CC offset + wrong start":     (90, -0.05),
}

print("=" * 62)
print(f"{'Method':35s} {'MAE %':>8s} {'Max error %':>14s}")
print("=" * 62)
for name, (start, offset) in scenarios.items():
    est = test_df.groupby("Source_File", group_keys=False).apply(
        lambda p: coulomb_count(p, start, offset)
    )
    err = (est - test_df["SOC_Percent"]).abs()
    print(f"{name:35s} {err.mean():8.2f} {err.max():14.2f}")

ml_err = (test_df["SOC_ML"] - test_df["SOC_Percent"]).abs()
print("-" * 62)
print(f"{'ML model (no start SOC needed)':35s} {ml_err.mean():8.2f} {ml_err.max():14.2f}")
print("=" * 62)
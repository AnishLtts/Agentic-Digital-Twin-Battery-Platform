# SOC predictor: the standard interface the Agentic AI will call
import os
import joblib
import numpy as np
import pandas as pd

MODEL_FILE = os.path.join(os.path.dirname(__file__), "soc_model.joblib")
SMOOTH_WINDOW = 60

# Measured test error (smoothed MAE) per temperature, from Step 5
TEMPS_C = [-20, -10, 0, 10, 25, 40]
MAE_PER_TEMP = [0.93, 1.02, 0.83, 0.78, 0.51, 0.25]

# Range of the training data
VOLTAGE_RANGE = (2.5, 4.2)
TEMP_RANGE = (-20, 45)

_saved = joblib.load(MODEL_FILE)
_model, _FEATURES = _saved["model"], _saved["features"]


def predict_soc(history: pd.DataFrame) -> dict:
    """
    history: the most recent readings at 1 Hz (ideally 120 s or more),
             with columns Voltage (V), Current (A), Temperature (C).
    Returns a dictionary the agent can read.
    """
    warnings = []
    if len(history) < 120:
        warnings.append(f"Only {len(history)} s of history; 120 s recommended")

    h = history[["Voltage", "Current", "Temperature"]].reset_index(drop=True).copy()
    for window in [30, 120]:
        for col in ["Voltage", "Current", "Temperature"]:
            h[f"{col}_avg_{window}s"] = h[col].rolling(window, min_periods=1).mean()

    raw = np.clip(_model.predict(h[_FEATURES]), 0, 100)
    soc = float(pd.Series(raw).rolling(SMOOTH_WINDOW, min_periods=1).mean().iloc[-1])

    v, t = h["Voltage"].iloc[-1], h["Temperature"].iloc[-1]
    if not VOLTAGE_RANGE[0] <= v <= VOLTAGE_RANGE[1]:
        warnings.append(f"Voltage {v:.3f} V is outside the training range")
    if not TEMP_RANGE[0] <= t <= TEMP_RANGE[1]:
        warnings.append(f"Temperature {t:.1f} C is outside the training range")

    uncertainty = float(np.interp(t, TEMPS_C, MAE_PER_TEMP))
    if warnings:
        uncertainty *= 2  # be less confident when inputs look unusual

    return {
        "model": "SOC_Estimator_v1",
        "soc_percent": round(soc, 2),
        "uncertainty_percent": round(uncertainty, 2),
        "confidence": "low" if warnings else "high",
        "warnings": warnings,
    }


# ---------- Self-test: run this file directly ----------
if __name__ == "__main__":
    df = pd.read_csv("demo/soc/soc_test_drives.csv") if os.path.exists(
        "demo/soc/soc_test_drives.csv") else pd.read_csv("data/soc/soc_dataset_1Hz.csv")
    drive = df[df["Source_File"] == "n20degC/611_Mixed8.csv"].reset_index(drop=True)

    print("Self-test on an unseen -20 C drive")
    print("=" * 60)
    for second in [300, 1500, 3000, len(drive) - 1]:
        window = drive.iloc[max(0, second - 300): second + 1]
        result = predict_soc(window)
        true_soc = drive["SOC_Percent"].iloc[second]
        print(f"t={second:5d}s  true={true_soc:6.2f}%  "
              f"predicted={result['soc_percent']:6.2f}% "
              f"± {result['uncertainty_percent']}%  ({result['confidence']})")

    print("\nExample output the agent receives:")
    print(predict_soc(drive.iloc[1200:1501]))
# Step 8: Robustness test - add realistic sensor noise and bias to the inputs
import joblib
import numpy as np
import pandas as pd

DATA_FILE = "data/soc/soc_dataset_1Hz.csv"
MODEL_FILE = "models/soc/soc_model.joblib"

saved = joblib.load(MODEL_FILE)
model, FEATURES = saved["model"], saved["features"]

df = pd.read_csv(DATA_FILE)
test_base = df[df["Source_File"].str.contains("Mixed8")].copy()
rng = np.random.default_rng(42)

# (voltage noise V, current noise A, temp noise C, voltage bias V)
scenarios = {
    "No noise (lab sensors)":        (0.000, 0.00, 0.0, 0.000),
    "Low noise (good BMS)":          (0.005, 0.05, 0.5, 0.000),
    "Medium noise (typical BMS)":    (0.010, 0.10, 1.0, 0.000),
    "High noise (cheap sensors)":    (0.020, 0.20, 2.0, 0.000),
    "Voltage bias +10 mV":           (0.000, 0.00, 0.0, 0.010),
    "Voltage bias -10 mV":           (0.000, 0.00, 0.0, -0.010),
}

rows = []
for name, (v_n, i_n, t_n, v_bias) in scenarios.items():
    test = test_base.copy()
    n = len(test)
    test["Voltage"] += rng.normal(0, v_n, n) + v_bias
    test["Current"] += rng.normal(0, i_n, n)
    test["Temperature"] += rng.normal(0, t_n, n)

    # Rebuild the averaged features from the noisy signals
    g = test.groupby("Source_File")
    for window in [30, 120]:
        for col in ["Voltage", "Current", "Temperature"]:
            test[f"{col}_avg_{window}s"] = g[col].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )

    pred = np.clip(model.predict(test[FEATURES]), 0, 100)
    test["Pred"] = pred
    test["Pred"] = test.groupby("Source_File")["Pred"].transform(
        lambda s: s.rolling(60, min_periods=1).mean()
    )
    err = (test["Pred"] - test["SOC_Percent"]).abs()
    rows.append({"Scenario": name,
                 "MAE %": round(err.mean(), 2),
                 "Max error %": round(err.max(), 2)})

summary = pd.DataFrame(rows)
print("=" * 60)
print("SENSOR NOISE ROBUSTNESS TEST (unseen Mixed8 drives)")
print("=" * 60)
print(summary.to_string(index=False))
summary.to_csv("outputs/soc/noise_test_results.csv", index=False)
print("\n✅ Saved to outputs/soc/noise_test_results.csv")
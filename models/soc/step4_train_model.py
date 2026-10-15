# Step 4: Train the SOC estimation model and test it on unseen drive cycles
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "data/soc/soc_dataset_1Hz.csv"
MODEL_FILE = "models/soc/soc_model.joblib"

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

# ---------- Feature engineering (per file, so files don't mix) ----------
groups = df.groupby("Source_File")
for window in [30, 120]:
    for col in ["Voltage", "Current", "Temperature"]:
        df[f"{col}_avg_{window}s"] = groups[col].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

FEATURES = [
    "Voltage", "Current", "Temperature",
    "Voltage_avg_30s", "Current_avg_30s", "Temperature_avg_30s",
    "Voltage_avg_120s", "Current_avg_120s", "Temperature_avg_120s",
]
TARGET = "SOC_Percent"

# ---------- Train / test split by file ----------
is_test = df["Source_File"].str.contains("Mixed8")
train_df = df[~is_test]
test_df = df[is_test]

print(f"Training files : {train_df['Source_File'].nunique()}  rows={len(train_df):,}")
print(f"Test files     : {test_df['Source_File'].nunique()}  rows={len(test_df):,}")

# ---------- Train ----------
print("\nTraining model... (about 1 minute)")
model = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=63, random_state=42
)
model.fit(train_df[FEATURES], train_df[TARGET])

# ---------- Evaluate on unseen files ----------
test_df = test_df.copy()
test_df["SOC_Predicted"] = np.clip(model.predict(test_df[FEATURES]), 0, 100)
errors = test_df["SOC_Predicted"] - test_df[TARGET]

mae = mean_absolute_error(test_df[TARGET], test_df["SOC_Predicted"])
rmse = np.sqrt(mean_squared_error(test_df[TARGET], test_df["SOC_Predicted"]))
r2 = r2_score(test_df[TARGET], test_df["SOC_Predicted"])

print("\n" + "=" * 60)
print("SOC MODEL RESULTS (unseen Mixed8 drive cycles)")
print("=" * 60)
print(f"MAE        : {mae:.2f} % SOC")
print(f"RMSE       : {rmse:.2f} % SOC")
print(f"Max error  : {errors.abs().max():.2f} % SOC")
print(f"R2 score   : {r2:.4f}")

print("\nError per temperature (MAE % SOC):")
per_temp = test_df.assign(abs_err=errors.abs()).groupby("Ambient_Temp_C")["abs_err"].mean()
print(per_temp.round(2))

# ---------- Save the model ----------
joblib.dump({"model": model, "features": FEATURES}, MODEL_FILE)
print(f"\n✅ Model saved to {MODEL_FILE}")

# ---------- Plot true vs predicted for two test files ----------
plot_files = ["25degC/552_Mixed8.csv", "n20degC/611_Mixed8.csv"]
fig, axes = plt.subplots(len(plot_files), 1, figsize=(12, 8))
for ax, name in zip(axes, plot_files):
    part = test_df[test_df["Source_File"] == name]
    ax.plot(part["Time_s"], part[TARGET], label="True SOC", color="green")
    ax.plot(part["Time_s"], part["SOC_Predicted"], label="Predicted SOC",
            color="purple", alpha=0.7)
    ax.set_title(f"SOC Estimation on unseen data: {name}")
    ax.set_ylabel("SOC (%)")
    ax.grid(True)
    ax.legend()
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("outputs/soc/soc_prediction.png")
print("✅ Graph saved to outputs/soc/soc_prediction.png")
plt.show()
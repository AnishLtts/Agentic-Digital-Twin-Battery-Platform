# Step 7: Leave-one-drive-type-out cross-validation
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_FILE = "data/soc/soc_dataset_1Hz.csv"

df = pd.read_csv(DATA_FILE)
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

results = []
for held_out in ["UDDS", "HWFET", "LA92", "US06", "Mixed"]:
    is_test = df["Drive_Cycle"] == held_out
    train_df, test_df = df[~is_test], df[is_test].copy()

    print(f"Training without {held_out}...")
    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=63, random_state=42
    )
    model.fit(train_df[FEATURES], train_df[TARGET])

    test_df["Pred"] = np.clip(model.predict(test_df[FEATURES]), 0, 100)
    test_df["Pred"] = test_df.groupby("Source_File")["Pred"].transform(
        lambda s: s.rolling(60, min_periods=1).mean()
    )
    err = (test_df["Pred"] - test_df[TARGET]).abs()
    results.append({
        "Unseen drive type": held_out,
        "Test files": test_df["Source_File"].nunique(),
        "MAE %": round(err.mean(), 2),
        "Max error %": round(err.max(), 2),
    })

summary = pd.DataFrame(results)
print("\n" + "=" * 60)
print("CROSS-VALIDATION: each drive type never seen in training")
print("=" * 60)
print(summary.to_string(index=False))
print("-" * 60)
print(f"Average MAE across all folds: {summary['MAE %'].mean():.2f} %")
summary.to_csv("outputs/soc/cross_validation_results.csv", index=False)
print("✅ Saved to outputs/soc/cross_validation_results.csv")
# Creates a small demo dataset (unseen test drives only) and requirements.txt
import os
from importlib.metadata import version
import pandas as pd

df = pd.read_csv("data/soc/soc_dataset_1Hz.csv")
demo = df[df["Source_File"].str.contains("Mixed8")].copy()

# Round values to keep the file small
demo = demo.round({"Time_s": 0, "Voltage": 4, "Current": 3,
                   "Temperature": 2, "SOC_Percent": 3})

os.makedirs("demo/soc", exist_ok=True)
demo.to_csv("demo/soc/soc_test_drives.csv", index=False)
size_mb = os.path.getsize("demo/soc/soc_test_drives.csv") / 1e6
print(f"✅ Demo data saved: {len(demo):,} rows, {size_mb:.1f} MB")

# requirements.txt with the exact versions used to train the model
packages = ["numpy", "pandas", "scikit-learn", "joblib", "streamlit"]
with open("requirements.txt", "w") as f:
    for p in packages:
        f.write(f"{p}=={version(p)}\n")
print("✅ requirements.txt created")
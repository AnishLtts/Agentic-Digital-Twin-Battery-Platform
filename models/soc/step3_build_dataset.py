# Step 3: Combine all drive-cycle files from all temperatures into one dataset
import glob
import os
import pandas as pd

DATA_DIR = "data/soc"
OUTPUT_FILE = "data/soc/soc_dataset_1Hz.csv"
NOMINAL_CAPACITY_AH = 3.0

TEMPERATURE_FOLDERS = {
    "n20degC": -20, "n10degC": -10, "0degC": 0,
    "10degC": 10, "25degC": 25, "40degC": 40,
}
DRIVE_CYCLE_KEYWORDS = ["UDDS", "HWFET", "LA92", "US06", "Mixed"]


def find_header_line(file_path):
    """Find the line number where the column names start."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("Time Stamp"):
                return i
    return None


def load_drive_cycle(file_path):
    header_line = find_header_line(file_path)
    if header_line is None:
        return None

    # Skip the info lines and the units line just below the header
    skip = list(range(header_line)) + [header_line + 1]
    df = pd.read_csv(file_path, skiprows=skip, low_memory=False)
    df = df[["Prog Time", "Voltage", "Current", "Temperature", "Capacity"]].dropna()

    prog_time = pd.to_timedelta(df["Prog Time"])
    df["Time_s"] = (prog_time - prog_time.iloc[0]).dt.total_seconds()

    # Downsample from about 10 Hz to 1 Hz
    df["Second"] = df["Time_s"].astype(int)
    df = df.groupby("Second").first().reset_index()

    df["SOC_Percent"] = 100 * (1 + df["Capacity"] / NOMINAL_CAPACITY_AH)
    return df[["Time_s", "Voltage", "Current", "Temperature", "SOC_Percent"]]


all_data = []
for folder, ambient_temp in TEMPERATURE_FOLDERS.items():
    for file_path in sorted(glob.glob(os.path.join(DATA_DIR, folder, "*.csv"))):
        file_name = os.path.basename(file_path)
        cycle = next((k for k in DRIVE_CYCLE_KEYWORDS if k in file_name), None)
        if cycle is None:
            continue  # skip charge, capacity, HPPC and other lab tests

        df = load_drive_cycle(file_path)
        if df is None or df.empty:
            print(f"⚠ Skipped (no data found): {folder}/{file_name}")
            continue

        soc_drop = df["SOC_Percent"].iloc[0] - df["SOC_Percent"].iloc[-1]
        if soc_drop < 5:
            print(f"⚠ Skipped (almost no discharge): {folder}/{file_name}")
            continue

        df["Ambient_Temp_C"] = ambient_temp
        df["Drive_Cycle"] = cycle
        df["Source_File"] = f"{folder}/{file_name}"
        all_data.append(df)

        print(f"✅ {folder:8s} {file_name:22s} rows={len(df):6d} "
              f"start V={df['Voltage'].iloc[0]:.2f} "
              f"SOC {df['SOC_Percent'].iloc[0]:.0f}% → {df['SOC_Percent'].iloc[-1]:.0f}%")

dataset = pd.concat(all_data, ignore_index=True)
dataset.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print(f"Files loaded  : {len(all_data)}")
print(f"Total rows    : {len(dataset):,}")
print("\nRows per temperature:")
print(dataset["Ambient_Temp_C"].value_counts().sort_index())
print(f"\n✅ Saved to {OUTPUT_FILE}")
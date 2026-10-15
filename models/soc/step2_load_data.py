# Step 2: Load one drive-cycle file, create the SOC label, and plot it
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = "data/soc/25degC/551_UDDS.csv"
NOMINAL_CAPACITY_AH = 3.0

# Line 28 = column names, line 29 = units (skip it), lines 0-27 = test info (skip)
rows_to_skip = list(range(28)) + [29]
df = pd.read_csv(FILE_PATH, skiprows=rows_to_skip)

# Remove the empty column created by the comma at the end of each line
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

# Keep only the columns we need
df = df[["Prog Time", "Voltage", "Current", "Temperature", "Capacity"]]

# Convert the test clock into seconds from the start
prog_time = pd.to_timedelta(df["Prog Time"])
df["Time_s"] = (prog_time - prog_time.iloc[0]).dt.total_seconds()

# Create the true SOC label
df["SOC_Percent"] = 100 * (1 + df["Capacity"] / NOMINAL_CAPACITY_AH)

print("Shape (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary:")
print(df[["Voltage", "Current", "Temperature", "SOC_Percent"]].describe())

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(df["Time_s"], df["Voltage"], color="blue")
axes[0].set_ylabel("Voltage (V)")
axes[1].plot(df["Time_s"], df["Current"], color="orange")
axes[1].set_ylabel("Current (A)")
axes[2].plot(df["Time_s"], df["Temperature"], color="red")
axes[2].set_ylabel("Temp (C)")
axes[3].plot(df["Time_s"], df["SOC_Percent"], color="green")
axes[3].set_ylabel("SOC (%)")
axes[3].set_xlabel("Time (s)")
for ax in axes:
    ax.grid(True)
fig.suptitle("LG HG2 - UDDS Drive Cycle at 25 C")
plt.tight_layout()
plt.savefig("outputs/soc/udds_25degC.png")
print("\nGraph saved to outputs/soc/udds_25degC.png")
plt.show()
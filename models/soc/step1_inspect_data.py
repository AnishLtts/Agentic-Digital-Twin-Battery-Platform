# Step 1: Look at the raw structure of one SOC drive-cycle file
file_path = "data/soc/25degC/551_UDDS.csv"

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for line_number in range(40):
        line = f.readline()
        print(f"{line_number:02d}: {line.strip()}")
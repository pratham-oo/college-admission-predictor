import pandas as pd
import os
import glob
import re

data_folder = "data/last_year"  # where your round CSVs are
all_files = glob.glob(os.path.join(data_folder, "*.csv"))
combined = []

for file in all_files:
    match = re.search(r"Round_(\d+)", file, re.IGNORECASE)
    if not match:
        continue
    round_num = int(match.group(1))
    df = pd.read_csv(file)
    df["Round"] = round_num
    combined.append(df)
    print(f"Loaded {file} -> Round {round_num}, {len(df)} rows")

combined_df = pd.concat(combined, ignore_index=True)
combined_df.to_csv("data/combined_all_rounds.csv", index=False)
print(f"\nSaved {len(combined_df)} rows to data/combined_all_rounds.csv")
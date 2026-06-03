import pandas as pd
import numpy as np
import os
import glob
import re

LAST_YEAR_FOLDER = "data/last_year"
CURRENT_R1_FILE = "data/current_year_round1.csv"
OUTPUT_PREDICTIONS = "data/predicted_rounds.csv"

def load_last_year_rounds(folder):
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    combined = []
    for file in all_files:
        match = re.search(r"Round_(\d+)", file, re.IGNORECASE)
        if not match:
            continue
        round_num = int(match.group(1))
        df = pd.read_csv(file)
        df["Round"] = round_num
        combined.append(df)
    if not combined:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    df_all = pd.concat(combined, ignore_index=True)
    df_all["General"] = pd.to_numeric(df_all["General"], errors="coerce")
    return df_all

def load_current_r1(filepath):
    df = pd.read_csv(filepath)
    rename_map = {
        "collegename": "College Name",
        "stream": "Stream",
        "medium": "Medium",
        "ReservationDetails": "Reservation Details",
        "choicecode": "Choice Code",
        "status": "Status"          # rename to match previous year
    }
    df = df.rename(columns=rename_map)
    df["General"] = pd.to_numeric(df["General"], errors="coerce")
    if "Round" in df.columns:
        df = df[df["Round"] == 1]
    else:
        df["Round"] = 1
    return df

def build_cutoff_progression(last_year_df):
    progression = {}
    # Include Status in grouping
    grouped = last_year_df.groupby(["College Name", "Stream", "Medium", "Status", "Reservation Details"])
    for key, group in grouped:
        valid = group[~group["General"].isna()].copy()
        valid = valid.sort_values("Round")
        if len(valid) < 2:
            continue
        prog = {}
        for _, row in valid.iterrows():
            prog[int(row["Round"])] = row["General"]
        progression[key] = prog
    return progression

def predict_round_for_score(progression, this_year_r1, student_score):
    if not progression or this_year_r1 is None:
        return None
    if student_score >= this_year_r1:
        return 1
    first_round = min(progression.keys())
    last_round = max(progression.keys())
    first_cutoff = progression[first_round]
    last_cutoff = progression[last_round]
    if student_score < last_cutoff:
        return None
    total_drop = first_cutoff - last_cutoff
    total_rounds = last_round - first_round
    if total_drop <= 0 or total_rounds <= 0:
        return None
    drop_per_round = total_drop / total_rounds
    points_needed = this_year_r1 - student_score
    rounds_needed = int((points_needed + drop_per_round - 1) // drop_per_round)
    predicted = first_round + rounds_needed
    return min(predicted, 18)

if __name__ == "__main__":
    print("Loading last year's rounds...")
    last_year = load_last_year_rounds(LAST_YEAR_FOLDER)
    print(f"Loaded {len(last_year)} rows.")

    print("Loading current year R1...")
    current_r1 = load_current_r1(CURRENT_R1_FILE)
    print(f"Loaded {len(current_r1)} rows.")

    print("Building cutoff progression from last year...")
    progression = build_cutoff_progression(last_year)
    print(f"Built progression for {len(progression)} combinations.")

    print("Preparing predictions...")
    predictions = []
    for _, row in current_r1.iterrows():
        key = (row["College Name"], row["Stream"], row["Medium"], row["Status"], row["Reservation Details"])
        if key not in progression:
            continue
        this_year_r1 = row["General"]
        if pd.isna(this_year_r1):
            continue
        predictions.append({
            "College Name": row["College Name"],
            "Stream": row["Stream"],
            "Medium": row["Medium"],
            "Status": row["Status"],
            "Reservation Details": row["Reservation Details"],
            "R1_this_year": this_year_r1,
            "last_year_progression": progression[key]
        })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Saved {len(pred_df)} combinations to {OUTPUT_PREDICTIONS}")
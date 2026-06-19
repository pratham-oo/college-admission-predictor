import pandas as pd
import numpy as np
import os
import glob
import re

LAST_YEAR_FOLDER = "data/last_year"
CURRENT_R1_FILE = "data/current_year_round1.csv"
CURRENT_R2_FILE = "data/current_year_round2.csv"      # NEW
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
        "status": "Status"
    }
    df = df.rename(columns=rename_map)
    df["General"] = pd.to_numeric(df["General"], errors="coerce")
    if "Round" in df.columns:
        df = df[df["Round"] == 1]
    else:
        df["Round"] = 1
    return df

def load_current_r2(filepath):
    df = pd.read_csv(filepath)
    rename_map = {
        "collegename": "College Name",
        "stream": "Stream",
        "medium": "Medium",
        "ReservationDetails": "Reservation Details",
        "choicecode": "Choice Code",
        "status": "Status"
    }
    df = df.rename(columns=rename_map)
    df["General"] = pd.to_numeric(df["General"], errors="coerce")
    if "Round" in df.columns:
        df = df[df["Round"] == 2]
    else:
        df["Round"] = 2
    return df

def build_cutoff_progression(last_year_df):
    progression = {}
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

def compute_last_year_drop_info(progression):
    """
    For each key, compute:
    - total_drop (R1 - R18)
    - drop_R1_to_R2 (R1 - R2, if available)
    - remaining_drop_per_round = (total_drop - drop_R1_to_R2) / (18 - 2)
    """
    drop_info = {}
    for key, prog in progression.items():
        if 1 not in prog:
            continue
        r1 = prog[1]
        rounds = sorted(prog.keys())
        r18 = prog[rounds[-1]]
        total_drop = r1 - r18
        r2 = prog.get(2, r1)  # if R2 missing, assume no drop between R1 and R2
        drop_r1_r2 = r1 - r2
        remaining_drop = total_drop - drop_r1_r2
        remaining_rounds = 18 - 2  # from R2 to R18 is 16 steps
        if remaining_drop > 0 and remaining_rounds > 0:
            remaining_drop_per_round = remaining_drop / remaining_rounds
        else:
            remaining_drop_per_round = 0
        drop_info[key] = {
            "r1": r1,
            "r2": r2,
            "r18": r18,
            "total_drop": total_drop,
            "drop_r1_r2": drop_r1_r2,
            "remaining_drop_per_round": remaining_drop_per_round
        }
    return drop_info

if __name__ == "__main__":
    print("Loading last year's rounds...")
    last_year = load_last_year_rounds(LAST_YEAR_FOLDER)
    print(f"Loaded {len(last_year)} rows.")

    print("Loading this year's Round 1...")
    current_r1 = load_current_r1(CURRENT_R1_FILE)
    print(f"Loaded {len(current_r1)} rows.")

    print("Loading this year's Round 2...")
    current_r2 = load_current_r2(CURRENT_R2_FILE)
    print(f"Loaded {len(current_r2)} rows.")

    print("Building cutoff progression from last year...")
    progression = build_cutoff_progression(last_year)
    print(f"Built progression for {len(progression)} combinations.")

    print("Computing last year's drop info...")
    drop_info = compute_last_year_drop_info(progression)
    print(f"Computed drop info for {len(drop_info)} combinations.")

    # Merge this year's R1 and R2 to get actual R2 cutoffs
    merge_keys = ["College Name", "Stream", "Medium", "Status", "Reservation Details"]
    curr_merged = current_r1.merge(current_r2, on=merge_keys, suffixes=("_R1", "_R2"))

    print("Generating predictions...")
    predictions = []
    for _, row in curr_merged.iterrows():
        key = (row["College Name"], row["Stream"], row["Medium"], row["Status"], row["Reservation Details"])
        if key not in drop_info:
            continue
        r1_this = row["General_R1"]
        r2_this = row["General_R2"]
        if pd.isna(r1_this) or pd.isna(r2_this):
            continue
        
        # Actual R2 is known
        pred = {2: int(round(r2_this))}
        
        # Use last year's remaining drop per round to predict R3..R18
        remaining_drop_per_round = drop_info[key]["remaining_drop_per_round"]
        for r in range(3, 19):
            predicted = r2_this - remaining_drop_per_round * (r - 2)
            pred[r] = int(round(max(0, predicted)))
        
        predictions.append({
            "College Name": row["College Name"],
            "Stream": row["Stream"],
            "Medium": row["Medium"],
            "Status": row["Status"],
            "Reservation Details": row["Reservation Details"],
            "R1_this_year": r1_this,
            "R2_this_year": r2_this,
            "Predicted_R2": pred[2],
            "Predicted_R3": pred.get(3),
            "Predicted_R4": pred.get(4),
            "Predicted_R5": pred.get(5),
            "Predicted_R6": pred.get(6),
            "Predicted_R7": pred.get(7),
            "Predicted_R8": pred.get(8),
            "Predicted_R9": pred.get(9),
            "Predicted_R10": pred.get(10),
            "Predicted_R11": pred.get(11),
            "Predicted_R12": pred.get(12),
            "Predicted_R13": pred.get(13),
            "Predicted_R14": pred.get(14),
            "Predicted_R15": pred.get(15),
            "Predicted_R16": pred.get(16),
            "Predicted_R17": pred.get(17),
            "Predicted_R18": pred.get(18),
        })

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Saved {len(pred_df)} combinations to {OUTPUT_PREDICTIONS}")

    # Optional: compare predicted R2 vs actual R2 (validation)
    if len(pred_df) > 0:
        avg_error = (pred_df["Predicted_R2"] - pred_df["R2_this_year"]).abs().mean()
        print(f"Average absolute error for R2 predictions: {avg_error:.1f} marks")
from flask import Flask, render_template, request
import pandas as pd
from rapidfuzz import process, fuzz
import numpy as np

app = Flask(__name__)

# Load predictions
df = pd.read_csv("data/predicted_rounds.csv")

# Ensure all predicted columns exist; if not, create with NaN
for r in range(2, 19):
    col = f"Predicted_R{r}"
    if col not in df.columns:
        df[col] = np.nan

# Get unique college names for autocomplete
all_colleges = sorted(df["College Name"].unique())

# Aliases for common college names
college_aliases = {
    "patkar": "S. S. & L. S. PATKAR COLLEGE",
    "bhavans": "BHAVAN'S COLLEGE, ANDHERI",
    "dhanukar": "DHANUKAR COLLEGE",
    # add more as needed
}

def predict_round_for_score(row, student_score):
    """
    Given a row from df (with R1_this_year and Predicted_R2..R18),
    return the earliest round (1..18) where predicted cutoff <= score,
    or None if never.
    """
    r1 = row.get("R1_this_year")
    if pd.isna(r1):
        return None
    if student_score >= r1:
        return 1
    # Check from R2 onwards
    for r in range(2, 19):
        col = f"Predicted_R{r}"
        if col in row and pd.notna(row[col]) and student_score >= row[col]:
            return r
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        college_input = request.form["college"].strip()
        stream_input = request.form["stream"]
        medium_input = request.form["medium"]
        status_input = request.form["status"]
        reservation_input = request.form["reservation"]
        try:
            score = float(request.form["score"])
        except ValueError:
            result = {"error": "Please enter a valid numeric score."}
            return render_template("index.html", result=result, colleges=all_colleges)
        
        # Find actual college name
        lower_input = college_input.lower()
        actual_college = college_aliases.get(lower_input)
        if not actual_college:
            match = process.extractOne(college_input, all_colleges, scorer=fuzz.WRatio, score_cutoff=75)
            if match:
                actual_college = match[0]
            else:
                suggestions = [c for c in all_colleges if college_input.lower() in c.lower()]
                if suggestions:
                    result = {"error": f"College '{college_input}' not found. Did you mean: {', '.join(suggestions[:5])}?"}
                else:
                    result = {"error": f"College '{college_input}' not found in our data. It may not participate in CAP."}
                return render_template("index.html", result=result, colleges=all_colleges)
        
        # Filter
        mask = (
            (df["College Name"] == actual_college) &
            (df["Stream"] == stream_input) &
            (df["Medium"] == medium_input) &
            (df["Status"] == status_input) &
            (df["Reservation Details"] == reservation_input)
        )
        row = df[mask]
        if row.empty:
            college_data = df[df["College Name"] == actual_college]
            streams = college_data["Stream"].unique()
            mediums = college_data["Medium"].unique()
            statuses = college_data["Status"].unique()
            reservations = college_data["Reservation Details"].unique()
            error_msg = f"No data for {actual_college} with Stream='{stream_input}', Medium='{medium_input}', Status='{status_input}', Reservation='{reservation_input}'."
            if len(streams) > 0:
                error_msg += f" Available streams: {', '.join(streams)}."
            if len(mediums) > 0:
                error_msg += f" Available mediums: {', '.join(mediums)}."
            if len(statuses) > 0:
                error_msg += f" Available statuses: {', '.join(statuses)}."
            if len(reservations) > 0:
                error_msg += f" Available reservations: {', '.join(reservations)}."
            result = {"error": error_msg}
        else:
            pred_round = predict_round_for_score(row.iloc[0], score)
            if pred_round:
                result = {
                    "college": actual_college,
                    "stream": stream_input,
                    "medium": medium_input,
                    "status": status_input,
                    "reservation": reservation_input,
                    "score": score,
                    "round": pred_round
                }
            else:
                result = {"error": f"Your score {score} is below the final predicted cutoff for {actual_college} ({status_input}). Unlikely to get admission in this category."}
    return render_template("index.html", result=result, colleges=all_colleges)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
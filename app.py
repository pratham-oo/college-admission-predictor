from flask import Flask, render_template, request
import pandas as pd
import ast
from rapidfuzz import process, fuzz

app = Flask(__name__)

df = pd.read_csv("data/predicted_rounds.csv")
df["last_year_progression"] = df["last_year_progression"].apply(ast.literal_eval)

all_colleges = df["College Name"].unique()

college_aliases = {
    "patkar": "S. S. & L. S. PATKAR COLLEGE",
    "bhavans": "BHAVAN'S COLLEGE, ANDHERI",
    "dhanukar": "DHANUKAR COLLEGE",
}

def predict_round(row, student_score):
    prog = row["last_year_progression"]
    this_year_r1 = row["R1_this_year"]
    if student_score >= this_year_r1:
        return 1
    first_round = min(prog.keys())
    last_round = max(prog.keys())
    first_cutoff = prog[first_round]
    last_cutoff = prog[last_round]
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
            return render_template("index.html", result=result, colleges=sorted(all_colleges))
        
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
                return render_template("index.html", result=result, colleges=sorted(all_colleges))
        
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
            pred_round = predict_round(row.iloc[0], score)
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
                result = {"error": f"Your score {score} is below the final cutoff from last year for {actual_college} ({status_input}). Unlikely to get admission in this category."}
    return render_template("index.html", result=result, colleges=sorted(all_colleges))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
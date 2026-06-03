import pandas as pd

# Load the combined last-year data (from your earlier combine step)
df = pd.read_csv("data/combined_all_rounds.csv")  # adjust path if needed
df["General"] = pd.to_numeric(df["General"], errors="coerce")

# Filter for a college from your sample
college = "SAMATA HIGH SCHOOL"
stream = "Arts"
medium = "Marathi"
res = "Remaining seats (Pure)"

mask = (
    (df["College Name"] == college) &
    (df["Stream"] == stream) &
    (df["Medium"] == medium) &
    (df["Reservation Details"] == res)
)
prog = df[mask].sort_values("Round")[["Round", "General"]]
print(prog)
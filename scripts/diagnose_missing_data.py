import pandas as pd

df = pd.read_csv("data/chennai_rainfall_clean.csv", parse_dates=["Date"])

check_dates = ["2015-11-30", "2015-12-01", "2015-12-02",
               "2021-11-06", "2021-11-07", "2021-11-11",
               "2023-12-02", "2023-12-03", "2023-12-04", "2023-12-05"]

for d in check_dates:
    day_data = df[df["Date"] == d]
    print(f"\n--- {d} ---")
    if day_data.empty:
        print("NO STATIONS REPORTED ANYTHING THIS DAY (missing from raw data)")
    else:
        print(day_data[["Station", "Rainfall"]].to_string(index=False))
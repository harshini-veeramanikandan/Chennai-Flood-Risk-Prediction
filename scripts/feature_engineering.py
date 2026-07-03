import pandas as pd

CLEAN_PATH = "data/chennai_rainfall_clean.csv"
FEATURES_PATH = "data/chennai_rainfall_features.csv"


def main():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])

    # --- Aggregate all stations into one citywide daily rainfall value ---
    # Using mean across whatever stations reported that day.
    daily = (
        df.groupby("Date")["Rainfall"]
        .mean()
        .reset_index()
        .rename(columns={"Rainfall": "rainfall_mm"})
    )

    # Fill in missing calendar days (no station reported = assume 0 rain,
    # a common and defensible assumption -- document this in your report)
    full_range = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    daily = daily.set_index("Date").reindex(full_range).rename_axis("Date").reset_index()
    daily["rainfall_mm"] = daily["rainfall_mm"].fillna(0)

    print("Daily citywide series shape:", daily.shape)
    print(daily.head())

    # --- Feature engineering ---
    daily["rainfall_3day_sum"] = daily["rainfall_mm"].rolling(3).sum()
    daily["rainfall_7day_sum"] = daily["rainfall_mm"].rolling(7).sum()
    daily["month"] = daily["Date"].dt.month
    # Chennai's main flood risk season: Northeast monsoon, Oct-Dec
    daily["is_monsoon_month"] = daily["month"].isin([10, 11, 12]).astype(int)

    # Drop the first 6 rows where rolling windows are incomplete (NaN)
    daily = daily.dropna(subset=["rainfall_3day_sum", "rainfall_7day_sum"]).reset_index(drop=True)

    print("\nFinal feature table shape:", daily.shape)
    print(daily.head(10))

    daily.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved feature table to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
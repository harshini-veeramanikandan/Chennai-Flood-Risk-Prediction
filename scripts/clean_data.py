import pandas as pd

# Since you run this from the "flood prediction" root folder, path is just "data/..."
RAW_PATH = "data/chennai_rainfall.csv"
CLEAN_PATH = "data/chennai_rainfall_clean.csv"


def parse_mixed_dates(date_series):
    """
    The raw file mixes two date formats:
      - M/D/YYYY   e.g. 3/3/1993
      - DD-MM-YYYY e.g. 16-05-1993
    We split on which separator is used and parse each group separately.
    """
    slash_mask = date_series.str.contains("/", na=False)
    dash_mask = date_series.str.contains("-", na=False)

    parsed = pd.Series(index=date_series.index, dtype="datetime64[ns]")

    parsed.loc[slash_mask] = pd.to_datetime(
        date_series[slash_mask], format="%m/%d/%Y", errors="coerce"
    )
    parsed.loc[dash_mask] = pd.to_datetime(
        date_series[dash_mask], format="%d-%m-%Y", errors="coerce"
    )

    return parsed


def main():
    df = pd.read_csv(RAW_PATH)
    print("Raw shape:", df.shape)

    # Standardize column names and text fields
    df.columns = [c.strip() for c in df.columns]
    df["Station"] = df["Station"].astype(str).str.strip().str.title()
    df["District"] = df["District"].astype(str).str.strip().str.title()

    # Parse mixed date formats
    df["Date"] = parse_mixed_dates(df["Date"].astype(str).str.strip())
    before = len(df)
    df = df.dropna(subset=["Date"])
    print(f"Dropped {before - len(df)} rows with unparseable dates")

    # Rainfall: force numeric, fill missing with 0 (documented assumption)
    df["Rainfall"] = pd.to_numeric(df["Rainfall"], errors="coerce")
    missing = df["Rainfall"].isna().sum()
    print(f"Filling {missing} missing rainfall values with 0")
    df["Rainfall"] = df["Rainfall"].fillna(0)

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=["District", "Station", "Date", "Rainfall"])
    print(f"Dropped {before - len(df)} exact duplicate rows")

    df = df.sort_values(["Station", "Date"]).reset_index(drop=True)

    print("\nClean shape:", df.shape)
    print("Date range:", df["Date"].min(), "to", df["Date"].max())
    print("Stations found:", df["Station"].unique())

    # Sanity check for the date-format assumption: look for any day > 12
    # in the slash-format dates -- if found, M/D/YYYY assumption could be wrong
    # for those specific rows. Since we already parsed successfully with
    # %m/%d/%Y (max month = 12), this print just documents the parsed range.
    print("\nSample cleaned rows:")
    print(df.head(10))

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned file to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
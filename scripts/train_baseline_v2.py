import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_PATH = "data/chennai_rainfall_features.csv"

def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])

    # --- Target: is TOMORROW a flood-risk day? ---
    # Still using the top-5% 3-day-rainfall proxy, but now we predict it
    # one day AHEAD of time using only past information.
    threshold = df["rainfall_3day_sum"].quantile(0.95)
    df["flood_risk_today"] = (df["rainfall_3day_sum"] >= threshold).astype(int)

    # Shift the label backward so each row's target = NEXT day's flood_risk
    df["flood_risk_tomorrow"] = df["flood_risk_today"].shift(-1)

    # --- Features: only information available UP TO today (no leakage) ---
    # rainfall_mm, rainfall_3day_sum, rainfall_7day_sum as of TODAY are all
    # fine here, because we're using them to predict TOMORROW, not today.
    feature_cols = ["rainfall_mm", "rainfall_3day_sum", "rainfall_7day_sum", "month", "is_monsoon_month"]
    X = df[feature_cols]
    y = df["flood_risk_tomorrow"]

    # Drop the last row (no "tomorrow" label available) and any NaNs
    valid = y.notna()
    X, y = X[valid], y[valid].astype(int)
    dates = df["Date"][valid]

    print("Class balance (tomorrow's flood risk):\n", y.value_counts())
    print("Percent flood-risk days: {:.2f}%".format(y.mean() * 100))

    # Time-based split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print("Test period:", dates.iloc[split_index], "to", dates.iloc[-1])

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Classification report (forecasting tomorrow, no leakage) ---")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
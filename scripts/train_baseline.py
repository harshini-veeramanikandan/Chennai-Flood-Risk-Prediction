import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

FEATURES_PATH = "data/chennai_rainfall_features.csv"

def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])

    # --- Build target variable ---
    # Label a day as flood-risk (1) if 3-day cumulative rainfall is in the
    # top 5% of all recorded values historically. This is a PROXY target
    # (rainfall-intensity based), not confirmed real flood event ground truth.
    # Document this clearly as a limitation in your report.
    threshold = df["rainfall_3day_sum"].quantile(0.95)
    df["flood_risk"] = (df["rainfall_3day_sum"] >= threshold).astype(int)

    print(f"3-day rainfall threshold for 'flood_risk' label: {threshold:.1f} mm")
    print("Class balance:\n", df["flood_risk"].value_counts())
    print("Percent flood-risk days: {:.2f}%".format(df["flood_risk"].mean() * 100))

    # --- Features and target ---
    feature_cols = ["rainfall_mm", "rainfall_3day_sum", "rainfall_7day_sum", "month", "is_monsoon_month"]
    X = df[feature_cols]
    y = df["flood_risk"]

    # Time-based split: train on earlier years, test on later years
    # (avoids leaking future info into training -- important, mention this
    # choice explicitly in interviews)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    print("Train period:", df["Date"].iloc[0], "to", df["Date"].iloc[split_index - 1])
    print("Test period:", df["Date"].iloc[split_index], "to", df["Date"].iloc[-1])

    # --- Baseline model ---
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n--- Classification report (baseline: Logistic Regression) ---")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    df.to_csv("data/chennai_rainfall_with_target.csv", index=False)
    print("\nSaved labeled dataset to data/chennai_rainfall_with_target.csv")


if __name__ == "__main__":
    main()
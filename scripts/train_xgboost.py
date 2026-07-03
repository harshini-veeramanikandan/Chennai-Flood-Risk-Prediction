import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

FEATURES_PATH = "data/chennai_rainfall_features.csv"

def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])

    # --- Same leakage-free target as before: tomorrow's flood risk ---
    threshold = df["rainfall_3day_sum"].quantile(0.95)
    df["flood_risk_today"] = (df["rainfall_3day_sum"] >= threshold).astype(int)
    df["flood_risk_tomorrow"] = df["flood_risk_today"].shift(-1)

    feature_cols = ["rainfall_mm", "rainfall_3day_sum", "rainfall_7day_sum", "month", "is_monsoon_month"]
    X = df[feature_cols]
    y = df["flood_risk_tomorrow"]

    valid = y.notna()
    X, y = X[valid], y[valid].astype(int)
    dates = df["Date"][valid]

    # Time-based split (same as before)
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Train size before SMOTE: {len(X_train)}")
    print("Train class balance before SMOTE:\n", y_train.value_counts())

    # --- Apply SMOTE only to training data (never touch test data) ---
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    print(f"\nTrain size after SMOTE: {len(X_train_sm)}")
    print("Train class balance after SMOTE:\n", y_train_sm.value_counts())

    # --- Train XGBoost ---
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_sm, y_train_sm)

    y_pred = model.predict(X_test)

    print("\n--- Classification report (XGBoost + SMOTE) ---")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    f2 = fbeta_score(y_test, y_pred, beta=2)
    print(f"\nF2-score (recall-weighted, the metric that matters most here): {f2:.3f}")

    # --- Feature importance (quick look, SHAP comes later) ---
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)


if __name__ == "__main__":
    main()
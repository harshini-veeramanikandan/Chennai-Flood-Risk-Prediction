"""
Trains the final XGBoost flood-risk model once and saves it to disk.
Run this once (and again anytime you update the data). The API loads
the saved model instead of retraining on every request.
"""

import pandas as pd
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

FEATURES_PATH = "data/chennai_rainfall_features.csv"
MODEL_PATH = "model/flood_model.joblib"

FEATURE_COLS = ["rainfall_mm", "rainfall_3day_sum", "rainfall_7day_sum", "month", "is_monsoon_month"]


def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])

    threshold_label = df["rainfall_3day_sum"].quantile(0.95)
    df["flood_risk_today"] = (df["rainfall_3day_sum"] >= threshold_label).astype(int)
    df["flood_risk_tomorrow"] = df["flood_risk_today"].shift(-1)

    X = df[FEATURE_COLS]
    y = df["flood_risk_tomorrow"]
    valid = y.notna()
    X, y = X[valid], y[valid].astype(int)

    split_index = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train_sm, y_train_sm)

    # Save model + threshold + feature importances together
    feature_importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))

    joblib.dump({
        "model": model,
        "feature_cols": FEATURE_COLS,
        "threshold_label": float(threshold_label),
        "feature_importances": feature_importances,
    }, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print("Feature importances:", feature_importances)


if __name__ == "__main__":
    main()
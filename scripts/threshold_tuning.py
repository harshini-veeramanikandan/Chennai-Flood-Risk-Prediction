import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score

FEATURES_PATH = "data/chennai_rainfall_features.csv"

def main():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])

    threshold_label = df["rainfall_3day_sum"].quantile(0.95)
    df["flood_risk_today"] = (df["rainfall_3day_sum"] >= threshold_label).astype(int)
    df["flood_risk_tomorrow"] = df["flood_risk_today"].shift(-1)

    feature_cols = ["rainfall_mm", "rainfall_3day_sum", "rainfall_7day_sum", "month", "is_monsoon_month"]
    X = df[feature_cols]
    y = df["flood_risk_tomorrow"]

    valid = y.notna()
    X, y = X[valid], y[valid].astype(int)

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train_sm, y_train_sm)

    # Get predicted PROBABILITIES instead of hard predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # Try several thresholds and compare
    print("Threshold sweep (lower threshold = more alarms, higher recall):\n")
    for t in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15]:
        y_pred_t = (y_proba >= t).astype(int)
        f2 = fbeta_score(y_test, y_pred_t, beta=2)
        report = classification_report(y_test, y_pred_t, output_dict=True)
        precision_1 = report["1"]["precision"]
        recall_1 = report["1"]["recall"]
        print(f"Threshold {t:.2f} | Recall(flood)={recall_1:.2f} | Precision(flood)={precision_1:.2f} | F2={f2:.3f}")

    # Pick a threshold that prioritizes recall reasonably -- adjust based on the sweep above
    chosen_threshold = 0.3
    y_pred_final = (y_proba >= chosen_threshold).astype(int)

    print(f"\n--- Final report at threshold={chosen_threshold} ---")
    print(classification_report(y_test, y_pred_final))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_final))


if __name__ == "__main__":
    main()
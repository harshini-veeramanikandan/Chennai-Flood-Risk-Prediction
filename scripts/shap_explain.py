import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

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

    # --- SHAP explainability ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 1. Global summary plot: which features matter most overall
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    print("Saved shap_summary.png -- shows overall feature importance + direction of effect")

    # 2. Explain one specific high-risk prediction (a concrete example for your report/demo)
    y_proba = model.predict_proba(X_test)[:, 1]
    top_risk_idx = y_proba.argmax()  # the day the model was MOST confident about flood risk
    top_risk_date = df["Date"][valid].iloc[split_index:].iloc[top_risk_idx]

    print(f"\nMost confident flood-risk prediction was for: {top_risk_date.date()}")
    print("Feature values on that day:")
    print(X_test.iloc[top_risk_idx])
    print(f"Predicted probability: {y_proba[top_risk_idx]:.3f}")

    plt.figure()
    shap.force_plot(
        explainer.expected_value, shap_values[top_risk_idx], X_test.iloc[top_risk_idx],
        matplotlib=True, show=False,
    )
    plt.tight_layout()
    plt.savefig("shap_single_prediction.png", dpi=150)
    print("Saved shap_single_prediction.png -- shows why THIS specific day was flagged high-risk")


if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Chennai Flood Risk Predictor", layout="centered")

FEATURES_PATH = "data/chennai_rainfall_features.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Date"])
    threshold_label = df["rainfall_3day_sum"].quantile(0.95)
    df["flood_risk_today"] = (df["rainfall_3day_sum"] >= threshold_label).astype(int)
    df["flood_risk_tomorrow"] = df["flood_risk_today"].shift(-1)
    return df, threshold_label


@st.cache_resource
def train_model(df):
    feature_cols = ["rainfall_mm", "rainfall_3day_sum", "rainfall_7day_sum", "month", "is_monsoon_month"]
    X = df[feature_cols]
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
    return model, feature_cols


def main():
    st.title("🌧️ Chennai Flood Risk Predictor")
    st.caption("Predicts next-day flood risk from rainfall patterns, using an XGBoost model trained on 30 years of Chennai rainfall data (1993-2023).")

    df, threshold_label = load_data()
    model, feature_cols = train_model(df)

    st.markdown("---")
    st.subheader("📊 Historical rainfall context")
    st.line_chart(df.set_index("Date")["rainfall_3day_sum"].tail(365))
    st.caption(f"3-day cumulative rainfall, last 365 days in dataset. Flood-risk threshold: {threshold_label:.1f} mm")

    st.markdown("---")
    st.subheader("🔮 Try a rainfall scenario")

    col1, col2 = st.columns(2)
    with col1:
        rainfall_today = st.slider("Rainfall today (mm)", 0.0, 300.0, 20.0)
        rainfall_3day = st.slider("3-day cumulative rainfall (mm)", 0.0, 500.0, 60.0)
    with col2:
        rainfall_7day = st.slider("7-day cumulative rainfall (mm)", 0.0, 700.0, 100.0)
        month = st.selectbox("Month", list(range(1, 13)), index=9)

    is_monsoon = 1 if month in [10, 11, 12] else 0

    input_df = pd.DataFrame([{
        "rainfall_mm": rainfall_today,
        "rainfall_3day_sum": rainfall_3day,
        "rainfall_7day_sum": rainfall_7day,
        "month": month,
        "is_monsoon_month": is_monsoon,
    }])

    proba = model.predict_proba(input_df[feature_cols])[0, 1]

    st.markdown("### Prediction")
    if proba >= 0.4:
        st.error(f"⚠️ HIGH flood risk tomorrow — predicted probability: {proba:.1%}")
    elif proba >= 0.2:
        st.warning(f"⚡ Moderate flood risk tomorrow — predicted probability: {proba:.1%}")
    else:
        st.success(f"✅ Low flood risk tomorrow — predicted probability: {proba:.1%}")

    st.progress(min(float(proba), 1.0))

    st.markdown("---")
    st.caption(
        "Note: Flood risk label is a rainfall-intensity proxy (top 5% of historical 3-day rainfall), "
        "not confirmed ground-truth flood event data. See project README for full methodology and limitations."
    )


if __name__ == "__main__":
    main()
"""
FastAPI backend for the Chennai Flood Risk Predictor.

Run with:
    uvicorn backend:app --reload --port 8000

Requires model/flood_model.joblib to exist -- run scripts/save_model.py first.
"""

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Chennai Flood Risk API")

# Allow the React dev server (localhost:5173 by default with Vite) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = joblib.load("model/flood_model.joblib")
model = bundle["model"]
feature_cols = bundle["feature_cols"]
threshold_label = bundle["threshold_label"]
feature_importances = bundle["feature_importances"]

rainfall_df = pd.read_csv("data/chennai_rainfall_features.csv", parse_dates=["Date"])

explainer = shap.TreeExplainer(model)


class PredictionInput(BaseModel):
    rainfall_mm: float
    rainfall_3day_sum: float
    rainfall_7day_sum: float
    month: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Chennai Flood Risk API is running"}


@app.get("/history")
def get_history(days: int = 365):
    """Returns recent daily rainfall for the frontend chart."""
    recent = rainfall_df.tail(days)
    return {
        "dates": recent["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "rainfall_3day_sum": recent["rainfall_3day_sum"].round(1).tolist(),
        "threshold": round(threshold_label, 1),
    }


@app.get("/feature-importance")
def get_feature_importance():
    """Returns the model's global feature importances for the explainability panel."""
    sorted_items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    return {"features": [{"name": k, "importance": round(v, 4)} for k, v in sorted_items]}


@app.get("/climatology")
def get_climatology():
    """Average 3-day cumulative rainfall per calendar month, across all years -- powers the monsoon calendar heatmap."""
    monthly = rainfall_df.groupby("month")["rainfall_3day_sum"].mean().round(1)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return {
        "months": [
            {"month": month_names[m - 1], "avg_rainfall": monthly.get(m, 0)}
            for m in range(1, 13)
        ]
    }


@app.post("/predict")
def predict(input: PredictionInput):
    is_monsoon_month = 1 if input.month in [10, 11, 12] else 0

    row = pd.DataFrame([{
        "rainfall_mm": input.rainfall_mm,
        "rainfall_3day_sum": input.rainfall_3day_sum,
        "rainfall_7day_sum": input.rainfall_7day_sum,
        "month": input.month,
        "is_monsoon_month": is_monsoon_month,
    }])

    proba = float(model.predict_proba(row[feature_cols])[0, 1])

    if proba >= 0.4:
        risk_label = "HIGH"
    elif proba >= 0.2:
        risk_label = "MODERATE"
    else:
        risk_label = "LOW"

    return {
        "probability": round(proba, 4),
        "risk_label": risk_label,
        "threshold_used": threshold_label,
    }


@app.post("/explain")
def explain(input: PredictionInput):
    """
    Live SHAP explanation for THIS specific scenario -- shows how each
    feature value pushed the prediction up or down from the model's
    baseline expectation. Computed fresh per request, not precomputed.
    """
    is_monsoon_month = 1 if input.month in [10, 11, 12] else 0
    row = pd.DataFrame([{
        "rainfall_mm": input.rainfall_mm,
        "rainfall_3day_sum": input.rainfall_3day_sum,
        "rainfall_7day_sum": input.rainfall_7day_sum,
        "month": input.month,
        "is_monsoon_month": is_monsoon_month,
    }])[feature_cols]

    shap_values = explainer.shap_values(row)[0]
    base_value = float(explainer.expected_value)

    contributions = [
        {"feature": feature_cols[i], "value": float(row.iloc[0, i]), "contribution": round(float(shap_values[i]), 4)}
        for i in range(len(feature_cols))
    ]
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    return {"base_value": round(base_value, 4), "contributions": contributions}
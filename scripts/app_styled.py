import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Chennai Flood Risk Predictor", layout="wide", page_icon="🌧️")

FEATURES_PATH = "data/chennai_rainfall_features.csv"

# ---------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #0C1A26 0%, #10202E 100%);
        color: #DCE6ED;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: #EAF1F6 !important;
        font-weight: 600 !important;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-size: 2.6rem;
        color: #EAF1F6;
        margin-bottom: 0.2rem;
        line-height: 1.1;
    }

    .hero-sub {
        color: #7E93A7;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        max-width: 600px;
    }

    .instrument-panel {
        background: #16283A;
        border: 1px solid #223A4E;
        border-radius: 10px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4FA8C9;
        margin-bottom: 0.4rem;
    }

    .metric-mono {
        font-family: 'IBM Plex Mono', monospace;
        color: #DCE6ED;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] > div > div {
        background: #4FA8C9 !important;
    }
    .stSlider [role="slider"] {
        background: #EAF1F6 !important;
        border: 2px solid #4FA8C9 !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background-color: #0C1A26 !important;
        border: 1px solid #223A4E !important;
    }

    hr {
        border-color: #223A4E !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# DATA + MODEL
# ---------------------------------------------------------------
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
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           eval_metric="logloss", random_state=42)
    model.fit(X_train_sm, y_train_sm)
    return model, feature_cols


def risk_gauge(proba):
    """Circular instrument-style gauge, like an analog rain gauge dial."""
    if proba >= 0.4:
        color = "#E85D4E"
        label = "HIGH"
    elif proba >= 0.2:
        color = "#E8A33D"
        label = "MODERATE"
    else:
        color = "#4FA8C9"
        label = "LOW"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 40, "color": "#EAF1F6"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#7E93A7", "tickfont": {"color": "#7E93A7"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#0C1A26",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "#152436"},
                {"range": [20, 40], "color": "#1B2C3E"},
                {"range": [40, 100], "color": "#22303F"},
            ],
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#DCE6ED"},
    )
    return fig, label, color


def rainfall_strip(df):
    """Seismograph/recording-strip styled rainfall chart."""
    recent = df.tail(365)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["Date"], y=recent["rainfall_3day_sum"],
        line=dict(color="#4FA8C9", width=1.6),
        fill="tozeroy", fillcolor="rgba(79,168,201,0.15)",
        name="3-day rainfall",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, color="#7E93A7", tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(showgrid=True, gridcolor="#1B2C3E", color="#7E93A7", tickfont=dict(family="IBM Plex Mono", size=10)),
        font=dict(color="#DCE6ED"),
    )
    return fig


# ---------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------
df, threshold_label = load_data()
model, feature_cols = train_model(df)

st.markdown('<div class="eyebrow">MONSOON INSTRUMENT · CHENNAI, TAMIL NADU</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Flood Risk Station</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">An XGBoost model reads 30 years of Chennai rainfall records (1993–2023) '
    'to forecast tomorrow\'s flood risk from today\'s rainfall pattern.</div>',
    unsafe_allow_html=True,
)

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<div class="instrument-panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">RAINFALL RECORD · LAST 365 DAYS</div>', unsafe_allow_html=True)
    st.plotly_chart(rainfall_strip(df), use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<span class="metric-mono" style="color:#7E93A7; font-size:0.8rem;">'
        f'FLOOD-RISK THRESHOLD: {threshold_label:.1f} MM (3-DAY CUMULATIVE)</span>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="instrument-panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">SET TOMORROW\'S SCENARIO</div>', unsafe_allow_html=True)
    rainfall_today = st.slider("Rainfall today (mm)", 0.0, 300.0, 20.0)
    rainfall_3day = st.slider("3-day cumulative rainfall (mm)", 0.0, 500.0, 60.0)
    rainfall_7day = st.slider("7-day cumulative rainfall (mm)", 0.0, 700.0, 100.0)
    month = st.selectbox("Month", list(range(1, 13)), index=9,
                          format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    is_monsoon = 1 if month in [10, 11, 12] else 0
    input_df = pd.DataFrame([{
        "rainfall_mm": rainfall_today,
        "rainfall_3day_sum": rainfall_3day,
        "rainfall_7day_sum": rainfall_7day,
        "month": month,
        "is_monsoon_month": is_monsoon,
    }])
    proba = model.predict_proba(input_df[feature_cols])[0, 1]
    fig, label, color = risk_gauge(proba)

    st.markdown('<div class="instrument-panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">FORECAST</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<div style="text-align:center; font-family:IBM Plex Mono; letter-spacing:0.1em; '
        f'color:{color}; font-size:1.1rem; margin-top:-1rem;">{label} RISK — TOMORROW</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="instrument-panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">METHODOLOGY NOTE</div>', unsafe_allow_html=True)
    st.markdown(
        '<span style="color:#7E93A7; font-size:0.85rem;">Flood risk is a rainfall-intensity proxy '
        '(top 5% of historical 3-day rainfall totals) — not confirmed ground-truth flood event data. '
        'See project README for full methodology and limitations.</span>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
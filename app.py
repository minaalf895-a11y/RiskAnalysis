"""
RiskAnalysis - Streamlit App
Matches the original desktop UI screenshot exactly (dark theme).
Run: streamlit run app.py
Place this file in the same folder as all .pkl files.
"""

import os
import math
import warnings
import streamlit as st
import joblib
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskAnalysis",
    page_icon="⬡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load artifacts
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required model file not found: {name}")
    return joblib.load(path)


@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    try:
        model        = _load("best_random_forest_model.pkl")
        scaler       = _load("scaler.pkl")
        feature_cols = _load("feature_columns.pkl")
        cap_values   = _load("cap_values.pkl")
        return model, scaler, feature_cols, cap_values, None
    except FileNotFoundError as e:
        return None, None, None, None, str(e)
    except Exception as e:
        return None, None, None, None, f"Unexpected error: {e}"


model, scaler, feature_cols, cap_values, load_error = load_artifacts()

IMPUTER_FILLS = {"MonthlyIncome": 5400.0, "NumberOfDependents": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Prediction pipeline
# ─────────────────────────────────────────────────────────────────────────────
def predict(inputs: dict) -> float:
    try:
        df = pd.DataFrame([inputs])[feature_cols]
        for col, cap in cap_values.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=float(cap))
        for col, fill in IMPUTER_FILLS.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill)
        arr  = scaler.transform(df)
        prob = model.predict_proba(arr)[0][1]
        return round(float(prob) * 100, 1)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions
# ─────────────────────────────────────────────────────────────────────────────
FIELDS = [
    ("RevolvingUtilizationOfUnsecuredLines", "REVOLVING UTILIZATION",       "💳",  1,
        lambda v: f"{v*100:.1f}%",       0.324,  0.0,    5.0,     0.001,
        "Total balance / total credit limits"),
    ("age",                                  "CREDIT HISTORY AGE",           "📅",  2,
        lambda v: f"{int(v)} Yrs",       52,     18,     100,     1,
        "Age of borrower in years"),
    ("NumberOfTime30-59DaysPastDueNotWorse", "30-59 DAYS DELINQUENCY",      "⚠️",  3,
        lambda v: f"{int(v)} Occur.",    1,      0,      20,      1,
        "Times 30-59 days past due"),
    ("DebtRatio",                            "DEBT RATIO",                   "📊",  4,
        lambda v: f"{v:.2f}",            0.38,   0.0,    50.0,    0.01,
        "Monthly debt / monthly gross income"),
    ("MonthlyIncome",                        "MONTHLY INCOME",               "💰",  5,
        lambda v: f"${v:,.0f}",          8420.0, 0.0,    50000.0, 100.0,
        "Monthly gross income in USD"),
    ("NumberOfOpenCreditLinesAndLoans",      "OPEN CREDIT LINES",            "🏦",  6,
        lambda v: f"{int(v)} Accounts",  12,     0,      60,      1,
        "Open loans and lines of credit"),
    ("NumberOfTimes90DaysLate",              "90+ DAYS LATE",                "🔴",  7,
        lambda v: f"{int(v)} Occur.",    0,      0,      20,      1,
        "Times 90+ days past due"),
    ("NumberRealEstateLoansOrLines",         "REAL ESTATE LOANS",            "🏠",  8,
        lambda v: f"{int(v)} Loan",      1,      0,      20,      1,
        "Mortgage and real estate loans"),
    ("NumberOfTime60-89DaysPastDueNotWorse", "60-89 DAYS DELINQUENCY",      "📋",  9,
        lambda v: f"{int(v)} Occur.",    0,      0,      20,      1,
        "Times 60-89 days past due"),
    ("NumberOfDependents",                   "NUMBER OF DEPENDENTS",         "👨‍👩‍👧", 10,
        lambda v: f"{int(v)}",           2,      0,      20,      1,
        "Dependents in family (excluding borrower)"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Gauge SVG  — dark bg, thin ring, matches screenshot
# ─────────────────────────────────────────────────────────────────────────────
def gauge_svg(prob: float) -> str:
    if prob >= 70:
        arc_color  = "#ef4444"
        risk_label = "HIGHRISK"
    elif prob >= 40:
        arc_color  = "#f59e0b"
        risk_label = "MEDIUMRISK"
    else:
        arc_color  = "#6c63ff"
        risk_label = "LOWRISK"

    cx, cy, R = 90, 90, 65
    start_deg  = -90
    sweep      = (prob / 100) * 359.9

    def polar(deg, r):
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    sx, sy = polar(start_deg, R)
    ex, ey = polar(start_deg + sweep, R)
    large  = 1 if sweep > 180 else 0
    arc    = f"M {sx:.3f} {sy:.3f} A {R} {R} 0 {large} 1 {ex:.3f} {ey:.3f}"

    return f"""<svg viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg"
     width="180" height="180" style="display:block;margin:0 auto;">
  <!-- track -->
  <circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="#2a2a2a" stroke-width="12"/>
  <!-- arc -->
  <path d="{arc}" fill="none" stroke="{arc_color}" stroke-width="12" stroke-linecap="round"/>
  <!-- dark center -->
  <circle cx="{cx}" cy="{cy}" r="{R - 14}" fill="#111111"/>
  <!-- percentage -->
  <text x="{cx}" y="{cy - 8}" text-anchor="middle"
        fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif"
        font-size="28" font-weight="700">{prob}%</text>
  <!-- label -->
  <text x="{cx}" y="{cy + 16}" text-anchor="middle"
        fill="#6b6b6b" font-family="'Segoe UI', Arial, sans-serif"
        font-size="9" letter-spacing="1">{risk_label}</text>
</svg>"""


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS — dark theme matching screenshot pixel-for-pixel
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── global dark background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] > div {
    background-color: #111111 !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"]        { background-color: #111111 !important; box-shadow: none !important; }
[data-testid="stToolbar"]       { display: none !important; }
[data-testid="stDecoration"]    { display: none !important; }
#MainMenu, footer               { visibility: hidden !important; }

.block-container {
    padding: 0 !important;
    max-width: 420px !important;
    margin: 0 auto !important;
    background: #111111 !important;
}

/* ── top bar ── */
.ra-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 20px 12px;
    background: #111111;
}
.ra-title {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 7px;
}
.ra-hex  { color: #6c63ff; font-size: 16px; }
.ra-menu { color: #555555; font-size: 22px; line-height: 1; }

/* ── gauge ── */
.ra-gauge {
    padding: 8px 0 4px;
    text-align: center;
    background: #111111;
}

/* ── heading ── */
.ra-main-title {
    font-size: 24px;
    font-weight: 800;
    color: #ffffff;
    padding: 10px 20px 0;
    background: #111111;
}
.ra-main-sub {
    font-size: 12px;
    color: #6b6b6b;
    padding: 6px 20px 16px;
    background: #111111;
    line-height: 1.65;
}

/* ── single-column factor cards ── */
.ra-cards {
    padding: 0 14px 8px;
    background: #111111;
}
.ra-card {
    display: flex;
    align-items: center;
    padding: 14px 14px 14px 13px;
    border: 1px solid #242424;
    border-radius: 16px;
    margin-bottom: 10px;
    background: #1a1a1a;
    position: relative;
    transition: border-color .15s;
}
.ra-card:hover { border-color: #333333; }

.ra-icon-wrap {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: #222233;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    margin-right: 15px;
}
.ra-card-body  { flex: 1; min-width: 0; }
.ra-card-label {
    font-size: 10px;
    font-weight: 600;
    color: #555555;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.ra-card-value {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.ra-factor-num {
    position: absolute;
    top: 11px;
    right: 13px;
    font-size: 9px;
    font-weight: 600;
    color: #333333;
    letter-spacing: 0.5px;
}

/* ── divider ── */
.ra-divider {
    border: none;
    border-top: 1px solid #242424;
    margin: 10px 14px 16px;
}

/* ── form heading ── */
.ra-form-head {
    font-size: 14px;
    font-weight: 700;
    color: #cccccc;
    padding: 0 14px 14px;
    background: #111111;
}

/* ── number input: dark style ── */
div[data-testid="stNumberInput"] {
    background: #111111 !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #666666 !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stNumberInput"] > div {
    background: #1a1a1a !important;
}
[data-testid="stNumberInput"] input {
    background-color: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    caret-color: #ffffff !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,.18) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] button {
    background-color: #252525 !important;
    border: none !important;
    color: #aaaaaa !important;
    border-radius: 7px !important;
}
[data-testid="stNumberInput"] button:hover {
    background-color: #333333 !important;
    color: #ffffff !important;
}

/* ── analyze button ── */
.stButton > button {
    background: #6c63ff !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 13px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    padding: 15px 24px !important;
    width: 100% !important;
    margin-top: 10px !important;
    cursor: pointer !important;
    transition: background .2s, transform .1s !important;
}
.stButton > button:hover  { background: #5248d0 !important; }
.stButton > button:active { transform: scale(.99) !important; }

/* ── streamlit alert overrides ── */
[data-testid="stAlert"] {
    background-color: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── hide any remaining white flashes ── */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.element-container,
.stMarkdown {
    background: transparent !important;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(CSS, unsafe_allow_html=True)

# ── Error gate ────────────────────────────────────────────────────────────────
if load_error:
    st.error(f"⚠️ **Model files not found**\n\n`{load_error}`")
    st.info(
        "Place these `.pkl` files in the same folder as `app.py`:\n\n"
        "- `best_random_forest_model.pkl`\n"
        "- `scaler.pkl`\n"
        "- `feature_columns.pkl`\n"
        "- `cap_values.pkl`"
    )
    st.stop()

# ── Session-state defaults ────────────────────────────────────────────────────
for col, _n, _i, _num, _fmt, default, *_rest in FIELDS:
    key = f"inp_{col}"
    if key not in st.session_state:
        st.session_state[key] = float(default)


def current_inputs() -> dict:
    out = {}
    for col, _name, _icon, _num, _fmt, default, mn, mx, _step, _help in FIELDS:
        raw = st.session_state.get(f"inp_{col}", default)
        try:
            val = float(raw)
            val = max(float(mn), min(float(mx), val))
        except (TypeError, ValueError):
            val = float(default)
        out[col] = val
    return out


inputs = current_inputs()
prob   = predict(inputs)

# ── Build top section HTML ────────────────────────────────────────────────────
page = ""

# Top bar
page += """
<div class="ra-topbar">
  <div class="ra-title"><span class="ra-hex">⬡</span>&nbsp;RISKANALYSIS</div>
  <div class="ra-menu">&#9776;</div>
</div>
"""

# Gauge
page += f'<div class="ra-gauge">{gauge_svg(prob)}</div>'

# Heading + subtext
page += """
<div class="ra-main-title">Probability of Default</div>
<div class="ra-main-sub">
  Comprehensive risk profile based on current<br>
  financial behavioral indicators and historical<br>
  data analysis.
</div>
"""

# Single-column factor cards
page += '<div class="ra-cards">'
for col, name, icon, num, fmt, default, mn, mx, step, _help in FIELDS:
    val_str = fmt(inputs[col])
    page += f"""<div class="ra-card">
  <div class="ra-factor-num">FACTOR {num:02d}</div>
  <div class="ra-icon-wrap">{icon}</div>
  <div class="ra-card-body">
    <div class="ra-card-label">{name}</div>
    <div class="ra-card-value">{val_str}</div>
  </div>
</div>
"""
page += "</div>"

st.markdown(page, unsafe_allow_html=True)

# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="ra-divider">', unsafe_allow_html=True)

# ── Form heading ──────────────────────────────────────────────────────────────
st.markdown('<div class="ra-form-head">✏️&nbsp;&nbsp;Edit Input Factors</div>',
            unsafe_allow_html=True)

# ── Input fields ──────────────────────────────────────────────────────────────
for col, name, icon, num, fmt, default, mn, mx, step, help_txt in FIELDS:
    if step == 1:
        st.number_input(
            f"{icon}  {name}",
            min_value=int(mn),
            max_value=int(mx),
            step=1,
            key=f"inp_{col}",
            help=help_txt,
        )
    else:
        fmt_str = "%.4f" if step < 0.01 else "%.2f"
        st.number_input(
            f"{icon}  {name}",
            min_value=float(mn),
            max_value=float(mx),
            step=float(step),
            format=fmt_str,
            key=f"inp_{col}",
            help=help_txt,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Analyze button ────────────────────────────────────────────────────────────
if st.button("🔍   Analyze Risk Profile", use_container_width=True):
    st.rerun()
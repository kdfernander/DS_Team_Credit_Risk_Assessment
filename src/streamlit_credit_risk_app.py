import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.preprocess import CORE_FEATURES, clean_features  # noqa: E402


APP_TITLE = "Credit Risk Assessment & Loan Default Prediction"
SCORING_MODEL_PATH = ROOT / "models" / "phase6_calibrated_scoring_model.joblib"
REASON_MODEL_PATH = ROOT / "models" / "phase6_reason_code_logreg.joblib"
APPLICATION_TEST_PATH = ROOT / "data" / "home-credit-default-risk" / "application_test.csv"
PREDICTIONS_PATH = ROOT / "reports" / "phase6_application_test_predictions.csv"
DATA_DICTIONARY_PATH = ROOT / "data_dictionary.csv"

INFERENCE_FEATURES = [col for col in CORE_FEATURES if col != "TARGET"]
DEMOGRAPHIC_COLUMNS = [
    "CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE", "NAME_CONTRACT_TYPE",
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE"
]

ORANGE = "#F97316"
ORANGE_DARK = "#C2410C"
ORANGE_LIGHT = "#FFEDD5"
AMBER = "#F59E0B"
SLATE = "#334155"
RED = "#EF4444"
GREEN = "#10B981"
DECISION_COLORS = {
    "APPROVE": GREEN,
    "REVIEW": AMBER,
    "DECLINE": RED,
}
RISK_TIER_COLORS = {
    "A": "#10B981",
    "B": "#84CC16",
    "C": "#F59E0B",
    "D": "#F97316",
    "E": "#EF4444",
}

FEATURE_LABELS = {
    "Applicant_ID": "Applicant ID",
    "PD": "Probability of Default (PD)",
    "Decision": "Decision",
    "Risk_Tier": "Risk Tier",
    "Reason_1": "Primary Reason",
    "Reason_2": "Secondary Reason",
    "Reason_3": "Third Reason",
    "SK_ID_CURR": "Applicant ID",
    "CODE_GENDER": "Applicant Gender",
    "CNT_CHILDREN": "Number of Children",
    "NAME_FAMILY_STATUS": "Family Status",
    "CNT_FAM_MEMBERS": "Number of Family Members",
    "NAME_EDUCATION_TYPE": "Highest Education Level",
    "DAYS_BIRTH": "Age in Days Before Application",
    "AGE": "Applicant Age in Years",
    "NAME_INCOME_TYPE": "Income Source Type",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer Organization Type",
    "DAYS_EMPLOYED": "Days Employed Before Application",
    "EMPLOYMENT_YEARS": "Employment History in Years",
    "AMT_INCOME_TOTAL": "Total Annual Income",
    "FLAG_OWN_CAR": "Owns a Car",
    "OWN_CAR_AGE": "Age of Applicant's Car",
    "FLAG_OWN_REALTY": "Owns Real Estate",
    "NAME_HOUSING_TYPE": "Housing Situation",
    "DAYS_REGISTRATION": "Days Since Current Address Registration",
    "DAYS_ID_PUBLISH": "Days Since ID Was Updated",
    "NAME_CONTRACT_TYPE": "Loan Contract Type",
    "AMT_CREDIT": "Loan Amount Requested",
    "AMT_ANNUITY": "Loan Annuity Amount",
    "AMT_GOODS_PRICE": "Goods Price",
    "DTI_PROXY": "Debt-to-Income Proxy",
    "LOAN_TO_INCOME": "Loan-to-Income Ratio",
    "EXT_SOURCE_1": "External Credit Score 1",
    "EXT_SOURCE_2": "External Credit Score 2",
    "EXT_SOURCE_3": "External Credit Score 3",
    "AMT_REQ_CREDIT_BUREAU_HOUR": "Credit Bureau Inquiries in Last Hour",
    "AMT_REQ_CREDIT_BUREAU_DAY": "Credit Bureau Inquiries in Last Day",
    "AMT_REQ_CREDIT_BUREAU_WEEK": "Credit Bureau Inquiries in Last Week",
    "AMT_REQ_CREDIT_BUREAU_MON": "Credit Bureau Inquiries in Last Month",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Credit Bureau Inquiries in Last Quarter",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit Bureau Inquiries in Last Year",
}

DEFAULT_GLOSSARY_ROWS = [
    ["CODE_GENDER", "Applicant gender", "Categorical", "Demographics"],
    ["CNT_CHILDREN", "Number of children", "Numeric", "Demographics"],
    ["NAME_FAMILY_STATUS", "Applicant family status", "Categorical", "Demographics"],
    ["CNT_FAM_MEMBERS", "Number of family members", "Numeric", "Demographics"],
    ["NAME_EDUCATION_TYPE", "Highest education level", "Categorical", "Demographics"],
    ["AGE", "Applicant age in years", "Numeric", "Risk factor"],
    ["NAME_INCOME_TYPE", "Type of income source", "Categorical", "Employment & Income"],
    ["OCCUPATION_TYPE", "Applicant occupation", "Categorical", "Employment & Income"],
    ["ORGANIZATION_TYPE", "Type of employer organization", "Categorical", "Employment & Income"],
    ["EMPLOYMENT_YEARS", "Employment history in years", "Numeric", "Employment history"],
    ["AMT_INCOME_TOTAL", "Total annual income", "Numeric", "Repayment capacity"],
    ["FLAG_OWN_CAR", "Whether applicant owns a car", "Categorical", "Stability"],
    ["OWN_CAR_AGE", "Age of applicant's car", "Numeric", "Stability"],
    ["FLAG_OWN_REALTY", "Whether applicant owns real estate", "Categorical", "Housing stability"],
    ["NAME_HOUSING_TYPE", "Type of housing situation", "Categorical", "Housing stability"],
    ["DAYS_REGISTRATION", "Days since registration at current address", "Numeric", "Stability"],
    ["DAYS_ID_PUBLISH", "Days since ID document was last updated", "Numeric", "Stability"],
    ["NAME_CONTRACT_TYPE", "Type of loan contract", "Categorical", "Loan characteristic"],
    ["AMT_CREDIT", "Loan amount requested", "Numeric", "Loan characteristic"],
    ["AMT_ANNUITY", "Loan annuity amount", "Numeric", "Loan repayment burden"],
    ["AMT_GOODS_PRICE", "Price of goods for which loan is given", "Numeric", "Loan characteristic"],
    ["DTI_PROXY", "Annuity divided by total income", "Numeric", "Debt burden"],
    ["LOAN_TO_INCOME", "Credit amount divided by total income", "Numeric", "Affordability"],
    ["EXT_SOURCE_1", "External credit risk score 1", "Numeric", "Credit risk signal"],
    ["EXT_SOURCE_2", "External credit risk score 2", "Numeric", "Credit risk signal"],
    ["EXT_SOURCE_3", "External credit risk score 3", "Numeric", "Credit risk signal"],
    ["AMT_REQ_CREDIT_BUREAU_HOUR", "Credit bureau inquiries in last hour", "Numeric", "Inquiry behavior"],
    ["AMT_REQ_CREDIT_BUREAU_DAY", "Credit bureau inquiries in last day", "Numeric", "Inquiry behavior"],
    ["AMT_REQ_CREDIT_BUREAU_WEEK", "Credit bureau inquiries in last week", "Numeric", "Inquiry behavior"],
    ["AMT_REQ_CREDIT_BUREAU_MON", "Credit bureau inquiries in last month", "Numeric", "Inquiry behavior"],
    ["AMT_REQ_CREDIT_BUREAU_QRT", "Credit bureau inquiries in last quarter", "Numeric", "Inquiry behavior"],
    ["AMT_REQ_CREDIT_BUREAU_YEAR", "Credit bureau inquiries in last year", "Numeric", "Inquiry behavior"],
]


@st.cache_resource
def load_models():
    scoring_model = joblib.load(SCORING_MODEL_PATH)
    reason_model = joblib.load(REASON_MODEL_PATH)
    return scoring_model, reason_model


@st.cache_data
def load_prediction_data():
    if not PREDICTIONS_PATH.exists():
        return None
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_application_test_data():
    if not APPLICATION_TEST_PATH.exists():
        return None
    return pd.read_csv(APPLICATION_TEST_PATH)


@st.cache_data
def load_glossary_data():
    if DATA_DICTIONARY_PATH.exists():
        glossary = pd.read_csv(DATA_DICTIONARY_PATH)
    else:
        glossary = pd.DataFrame(DEFAULT_GLOSSARY_ROWS, columns=["Column", "Description", "Type", "Used For"])

    glossary["Display Name"] = glossary["Column"].map(lambda x: FEATURE_LABELS.get(x, x.replace("_", " ").title()))
    return glossary[["Display Name", "Description", "Type", "Used For"]]



def inject_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #fff7ed 0%, #ffffff 22%);
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3 {{
            color: {ORANGE_DARK};
        }}
        [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 1rem;
            font-weight: 600;
        }}
        [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        [data-baseweb="tab"] {{
            background-color: #fff7ed;
            border-radius: 10px 10px 0 0;
            padding: 0.5rem 1rem;
        }}
        [aria-selected="true"] {{
            background-color: {ORANGE_LIGHT} !important;
            border-bottom: 3px solid {ORANGE};
        }}
        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid #fed7aa;
            padding: 1rem;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(249, 115, 22, 0.08);
        }}
        div[data-testid="stForm"] {{
            background: rgba(255,255,255,0.9);
            border: 1px solid #fed7aa;
            padding: 1rem 1rem 0.5rem 1rem;
            border-radius: 16px;
        }}
        .insight-card {{
            background: white;
            border-left: 6px solid {ORANGE};
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 4px 14px rgba(249, 115, 22, 0.08);
            margin-bottom: 0.75rem;
        }}
        .insight-title {{
            color: {ORANGE_DARK};
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .decision-pill {{
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            font-weight: 700;
            color: white;
            margin-right: 0.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



def labelize(value: str) -> str:
    return FEATURE_LABELS.get(value, value.replace("_", " ").title())



def rename_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = [labelize(col) for col in renamed.columns]
    return renamed


def normalize_reason_text(
    reason: str,
    decision: str | None = None,
    risk_tier: str | None = None
) -> str:
    if pd.isna(reason):
        return reason

    text = str(reason)

    replacements = {
        "AMT_GOODS_PRICE": "Goods Price",
        "AMT_CREDIT": "Loan Amount Requested",
        "AMT_ANNUITY": "Loan Annuity Amount",
        "AMT_INCOME_TOTAL": "Total Annual Income",
        "LOAN_TO_INCOME": "Loan-to-Income Ratio",
        "DTI_PROXY": "Debt-to-Income Proxy",
        "AGE": "Applicant Age",
        "EMPLOYMENT_YEARS": "Employment History",
        "OWN_CAR_AGE": "Car Age",
        "EXT_SOURCE_1": "External Credit Score 1",
        "EXT_SOURCE_2": "External Credit Score 2",
        "EXT_SOURCE_3": "External Credit Score 3",
        "DAYS_ID_PUBLISH": "Days Since ID Was Updated",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # normalize grammar first
    if "contributed lower risk" in text:
        text = text.replace("contributed lower risk", "contributed to lower risk")
    if "contributed higher risk" in text:
        text = text.replace("contributed higher risk", "contributed to higher risk")

    # decision-specific override
    if decision == "DECLINE" and "Goods Price contributed to lower risk" in text:
        text = text.replace(
            "Goods Price contributed to lower risk",
            "Goods Price contributed to higher risk"
        )

    # tier-specific override for Tier A
    if risk_tier == "A":
        tier_a_replacements = {
            "Younger applicant age": "Older applicant age",
            "Younger Applicant Age": "Older Applicant Age",
            "High loan-to-income ratio": "Low loan-to-income ratio",
            "High Loan-to-Income Ratio": "Low Loan-to-Income Ratio",
            "Short employment history": "Long employment history",
            "Shorter Employment History": "Long Employment History",
            "High annuity burden": "Low annuity burden",
            "High Loan Annuity Amount": "Low annuity burden",
        }
        for old, new in tier_a_replacements.items():
            text = text.replace(old, new)

    return text



def get_decision(pd_score: float) -> str:
    if pd_score < 0.05:
        return "APPROVE"
    if pd_score <= 0.12:
        return "REVIEW"
    return "DECLINE"



def get_risk_tier(pd_score: float) -> str:
    if pd_score < 0.03:
        return "A"
    if pd_score < 0.07:
        return "B"
    if pd_score < 0.12:
        return "C"
    if pd_score < 0.20:
        return "D"
    return "E"



def build_input_dataframe(form_values: dict) -> pd.DataFrame:
    row = {feature: form_values.get(feature, np.nan) for feature in INFERENCE_FEATURES}

    defaults = {
        "EXT_SOURCE_1": 0.50,
        "EXT_SOURCE_2": 0.50,
        "EXT_SOURCE_3": 0.50,
        "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
        "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
        "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 0.0,
    }
    for key, value in defaults.items():
        row.setdefault(key, value)

    df = pd.DataFrame([row])
    df = clean_features(df)
    return df



def prettify_reason(feature_name: str, sign: float) -> str:
    name = feature_name.replace("num__", "").replace("cat__", "")
    name = name.replace("onehot__", "")
    name = name.replace("imputer__", "")

    positive_reason_map = {
        "AMT_CREDIT": "High Loan Amount Requested",
        "AMT_ANNUITY": "High Loan Annuity Amount",
        "AMT_INCOME_TOTAL": "Income Level Increased Risk",
        "AMT_GOODS_PRICE": "Goods Price contributed to higher risk",
        "EXT_SOURCE_1": "External Credit Score 1 influenced risk",
        "EXT_SOURCE_2": "External Credit Score 2 influenced risk",
        "EXT_SOURCE_3": "External Credit Score 3 influenced risk",
        "LOAN_TO_INCOME": "High Loan-to-Income Ratio",
        "DTI_PROXY": "High Debt-to-Income Proxy",
        "AGE": "Younger Applicant Age",
        "EMPLOYMENT_YEARS": "Shorter Employment History",
        "OWN_CAR_AGE": "Car Age influenced the score",
        "AMT_REQ_CREDIT_BUREAU_HOUR": "Recent Credit Bureau Inquiries influenced risk",
        "AMT_REQ_CREDIT_BUREAU_DAY": "Recent Credit Bureau Inquiries influenced risk",
        "AMT_REQ_CREDIT_BUREAU_WEEK": "Recent Credit Bureau Inquiries influenced risk",
        "AMT_REQ_CREDIT_BUREAU_MON": "Recent Credit Bureau Inquiries influenced risk",
        "AMT_REQ_CREDIT_BUREAU_QRT": "Recent Credit Bureau Inquiries influenced risk",
        "AMT_REQ_CREDIT_BUREAU_YEAR": "Recent Credit Bureau Inquiries influenced risk",
    }
    lower_risk_map = {
        "AMT_CREDIT": "Loan Amount Requested contributed to lower risk",
        "AMT_ANNUITY": "Loan Annuity Amount contributed to lower risk",
        "AMT_INCOME_TOTAL": "Total Annual Income contributed to lower risk",
        "AMT_GOODS_PRICE": "Goods Price contributed to lower risk",
        "EXT_SOURCE_1": "External Credit Score 1 contributed to lower risk",
        "EXT_SOURCE_2": "External Credit Score 2 contributed to lower risk",
        "EXT_SOURCE_3": "External Credit Score 3 contributed to lower risk",
        "LOAN_TO_INCOME": "Loan-to-Income Ratio contributed to lower risk",
        "DTI_PROXY": "Debt-to-Income Proxy contributed to lower risk",
        "AGE": "Applicant Age contributed to lower risk",
        "EMPLOYMENT_YEARS": "Employment History contributed to lower risk",
        "OWN_CAR_AGE": "Car Age contributed to lower risk",
    }

    for key, value in positive_reason_map.items():
        if key in name:
            return value if sign >= 0 else lower_risk_map.get(key, f"{labelize(key)} contributed to lower risk")

    if "NAME_CONTRACT_TYPE" in name:
        return "Loan Contract Type influenced the score"
    if "NAME_INCOME_TYPE" in name:
        return "Income Source Type influenced the score"
    if "NAME_EDUCATION_TYPE" in name:
        return "Highest Education Level influenced the score"
    if "NAME_HOUSING_TYPE" in name:
        return "Housing Situation influenced the score"
    if "FLAG_OWN_REALTY" in name:
        return "Real Estate Ownership influenced the score"
    if "FLAG_OWN_CAR" in name:
        return "Car Ownership influenced the score"
    if "CODE_GENDER" in name:
        return "Applicant Gender influenced the score"
    if "OCCUPATION_TYPE" in name:
        return "Occupation influenced the score"
    if "ORGANIZATION_TYPE" in name:
        return "Employer Organization Type influenced the score"
    if "NAME_FAMILY_STATUS" in name:
        return "Family Status influenced the score"

    return f"{labelize(name)} influenced the score"



def generate_reason_codes(reason_pipeline, sample: pd.DataFrame, top_n: int = 3) -> list[str]:
    preprocess = reason_pipeline.named_steps["preprocess"]
    model = reason_pipeline.named_steps["model"]

    transformed = preprocess.transform(sample)
    feature_names = preprocess.get_feature_names_out()
    coefs = model.coef_[0]

    if hasattr(transformed, "toarray"):
        transformed_row = transformed.toarray()[0]
    else:
        transformed_row = np.asarray(transformed)[0]

    contributions = transformed_row * coefs
    ranked_idx = np.argsort(np.abs(contributions))[::-1]

    reasons = []
    seen = set()
    for idx in ranked_idx:
        if transformed_row[idx] == 0:
            continue
        text = prettify_reason(feature_names[idx], contributions[idx])
        if text not in seen:
            reasons.append(text)
            seen.add(text)
        if len(reasons) == top_n:
            break

    while len(reasons) < top_n:
        reasons.append("No additional strong driver identified")

    return reasons



def score_application(scoring_model, reason_model, application_df: pd.DataFrame) -> dict:
    pd_score = float(scoring_model.predict_proba(application_df)[0, 1])
    decision = get_decision(pd_score)
    risk_tier = get_risk_tier(pd_score)
    reasons = [
        normalize_reason_text(x, decision, risk_tier)
        for x in generate_reason_codes(reason_model, application_df, top_n=3)
    ]

    return {
        "pd": pd_score,
        "decision": decision,
        "risk_tier": risk_tier,
        "reasons": reasons,
    }



def merge_predictions_with_inputs(pred_df: pd.DataFrame | None, app_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if pred_df is None or app_df is None:
        return None

    merged = pred_df.copy()
    app_subset = app_df.copy()

    if "Applicant_ID" in merged.columns and "SK_ID_CURR" in app_subset.columns:
        merged = merged.merge(app_subset, left_on="Applicant_ID", right_on="SK_ID_CURR", how="left")

    for col in ["Reason_1", "Reason_2", "Reason_3"]:
        if col in merged.columns:
            merged[col] = merged.apply(
                lambda row: normalize_reason_text(
                    row[col],
                    row.get("Decision"),
                    row.get("Risk_Tier")
                ),
                axis=1
            )

    return merged



def reason_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    reason_cols = [col for col in ["Reason_1", "Reason_2", "Reason_3"] if col in df.columns]
    if not reason_cols:
        return pd.DataFrame(columns=["Reason", "Count"])

    stacked = df[reason_cols].stack().reset_index(drop=True)
    summary = stacked.value_counts().reset_index()
    summary.columns = ["Reason", "Count"]
    return summary



def reason_summary_by_decision(df: pd.DataFrame) -> pd.DataFrame:
    reason_cols = [col for col in ["Reason_1", "Reason_2", "Reason_3"] if col in df.columns]
    if not reason_cols or "Decision" not in df.columns:
        return pd.DataFrame(columns=["Decision", "Reason", "Count"])

    frames = []
    for col in reason_cols:
        temp = df[["Decision", col]].rename(columns={col: "Reason"}).dropna()
        frames.append(temp)

    long_df = pd.concat(frames, ignore_index=True)
    summary = long_df.groupby(["Decision", "Reason"]).size().reset_index(name="Count")
    return summary.sort_values(["Decision", "Count"], ascending=[True, False])



def reason_summary_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    reason_cols = [col for col in ["Reason_1", "Reason_2", "Reason_3"] if col in df.columns]
    if not reason_cols or "Risk_Tier" not in df.columns:
        return pd.DataFrame(columns=["Risk_Tier", "Reason", "Count"])

    frames = []
    for col in reason_cols:
        temp = df[["Risk_Tier", col]].rename(columns={col: "Reason"}).dropna()
        frames.append(temp)

    long_df = pd.concat(frames, ignore_index=True)
    summary = long_df.groupby(["Risk_Tier", "Reason"]).size().reset_index(name="Count")
    return summary.sort_values(["Risk_Tier", "Count"], ascending=[True, False])



def demographic_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns or "Decision" not in df.columns:
        return pd.DataFrame(columns=[group_col, "Decision", "Count"])

    summary = df.groupby([group_col, "Decision"]).size().reset_index(name="Count")
    return summary.sort_values("Count", ascending=False)



def make_decision_chart(df: pd.DataFrame):
    counts = df["Decision"].value_counts().rename_axis("Decision").reset_index(name="Count")
    fig = px.bar(counts, x="Decision", y="Count", color="Decision", color_discrete_map=DECISION_COLORS, text="Count")
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_traces(marker_line_width=0)
    return fig



def make_risk_chart(df: pd.DataFrame):
    counts = df["Risk_Tier"].value_counts().sort_index().rename_axis("Risk_Tier").reset_index(name="Count")
    fig = px.bar(counts, x="Risk_Tier", y="Count", color="Risk_Tier", color_discrete_map=RISK_TIER_COLORS, text="Count")
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_traces(marker_line_width=0)
    return fig



def make_reason_chart(df: pd.DataFrame, color_col: str | None = None):
    if color_col and color_col in df.columns:
        color_map = RISK_TIER_COLORS if color_col == "Risk_Tier" else DECISION_COLORS
        fig = px.bar(df, x="Count", y="Reason", color=color_col, orientation="h", color_discrete_map=color_map, text="Count")
    else:
        fig = px.bar(df, x="Count", y="Reason", orientation="h", color_discrete_sequence=[ORANGE], text="Count")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", yaxis={"categoryorder": "total ascending"})
    fig.update_traces(marker_line_width=0)
    return fig



def make_demographic_chart(df: pd.DataFrame, group_col: str):
    fig = px.bar(df, x=group_col, y="Count", color="Decision", barmode="group", color_discrete_map=DECISION_COLORS, text="Count")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_title=labelize(group_col), yaxis_title="Count")
    fig.update_traces(marker_line_width=0)
    return fig



def make_tier_donut(df: pd.DataFrame):
    counts = df["Risk_Tier"].value_counts().sort_index().rename_axis("Risk_Tier").reset_index(name="Count")
    fig = px.pie(counts, names="Risk_Tier", values="Count", hole=0.58, color="Risk_Tier", color_discrete_map=RISK_TIER_COLORS)
    fig.update_layout(paper_bgcolor="white")
    return fig



def render_insight_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">{title}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def display_pd_definition():
    render_insight_card(
        "What PD Means",
        "PD stands for <b>Probability of Default</b>. It is a number between 0 and 1 that estimates how likely an applicant is to default within the selected time window. For example, a PD of 0.08 means about an 8% estimated chance of default.",
    )



def display_overview_section(df: pd.DataFrame):
    st.subheader("Portfolio Overview")
    display_pd_definition()

    total_apps = len(df)
    approve_count = int((df["Decision"] == "APPROVE").sum()) if "Decision" in df.columns else 0
    review_count = int((df["Decision"] == "REVIEW").sum()) if "Decision" in df.columns else 0
    decline_count = int((df["Decision"] == "DECLINE").sum()) if "Decision" in df.columns else 0
    avg_pd = float(df["PD"].mean()) if "PD" in df.columns else 0.0
    most_common_tier = df["Risk_Tier"].value_counts().idxmax() if "Risk_Tier" in df.columns else "N/A"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Applications", total_apps)
    c2.metric("Approve", approve_count)
    c3.metric("Review", review_count)
    c4.metric("Decline", decline_count)
    c5.metric("Average PD", f"{avg_pd:.4f}")
    c6.metric("Most Common Tier", most_common_tier)

    insight1, insight2 = st.columns(2)
    top_decision = df["Decision"].value_counts().idxmax()
    top_reason = reason_summary_table(df).head(1)
    with insight1:
        render_insight_card("Largest Decision Segment", f"Most applications fall into the <b>{top_decision}</b> group.")
    with insight2:
        reason_text = top_reason.iloc[0]["Reason"] if not top_reason.empty else "No reason available"
        render_insight_card("Most Frequent Reason Code", f"The most common driver across predictions is <b>{reason_text}</b>.")

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        if "Decision" in df.columns:
            st.write("**Decision Distribution**")
            st.plotly_chart(make_decision_chart(df), use_container_width=True)
    with chart_col2:
        if "Risk_Tier" in df.columns:
            st.write("**Risk Tier Distribution**")
            st.plotly_chart(make_risk_chart(df), use_container_width=True)
    with chart_col3:
        if "Risk_Tier" in df.columns:
            st.write("**Risk Tier Mix**")
            st.plotly_chart(make_tier_donut(df), use_container_width=True)

    st.write("**Contributing Factors by Risk Tier**")
    tier_summary = reason_summary_by_tier(df)
    selected_tier = st.selectbox("View Tier Segment", ["A", "B", "C", "D", "E"], key="tier_factor_filter")
    tier_view = tier_summary[tier_summary["Risk_Tier"] == selected_tier].head(8)
    left, right = st.columns(2)
    with left:
        st.dataframe(rename_display_columns(tier_view), use_container_width=True, hide_index=True)
    with right:
        if not tier_view.empty:
            st.plotly_chart(make_reason_chart(tier_view, "Risk_Tier"), use_container_width=True)



def display_prediction_browser(df: pd.DataFrame):
    st.subheader("Application Test Predictions")

    left, right = st.columns([1, 3])
    with left:
        decision_options = ["All"] + sorted(df["Decision"].dropna().unique().tolist()) if "Decision" in df.columns else ["All"]
        selected_decision = st.selectbox("Filter by Decision", decision_options)
        min_pd = 0.0
        max_pd = 1.0
        if "PD" in df.columns:
            min_pd, max_pd = st.slider("PD Range", 0.0, 1.0, (0.0, 1.0), 0.01)

    filtered = df.copy()
    if selected_decision != "All" and "Decision" in filtered.columns:
        filtered = filtered[filtered["Decision"] == selected_decision]
    if "PD" in filtered.columns:
        filtered = filtered[(filtered["PD"] >= min_pd) & (filtered["PD"] <= max_pd)]

    display_cols = [
        col for col in [
            "Applicant_ID", "PD", "Decision", "Risk_Tier",
            "Reason_1", "Reason_2", "Reason_3",
            "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
            "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"
        ] if col in filtered.columns
    ]

    with right:
        styled = filtered[display_cols].copy()
        if "PD" in styled.columns:
            styled["PD"] = styled["PD"].round(4)
        st.dataframe(
            rename_display_columns(styled.sort_values("PD", ascending=False) if "PD" in styled.columns else styled),
            use_container_width=True,
            hide_index=True,
        )



def display_reason_analysis(df: pd.DataFrame):
    st.subheader("Reason Code Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Top Reasons Overall**")
        overall = reason_summary_table(df).head(10)
        st.dataframe(rename_display_columns(overall), use_container_width=True, hide_index=True)
        if not overall.empty:
            st.plotly_chart(make_reason_chart(overall), use_container_width=True)

    with col2:
        st.write("**Top Reasons by Decision**")
        by_decision = reason_summary_by_decision(df)
        selected_decision = st.selectbox("View Decision Segment", ["APPROVE", "REVIEW", "DECLINE"], key="reason_decision_filter")
        decision_view = by_decision[by_decision["Decision"] == selected_decision].head(10)
        st.dataframe(rename_display_columns(decision_view), use_container_width=True, hide_index=True)
        if not decision_view.empty:
            st.plotly_chart(make_reason_chart(decision_view, "Decision"), use_container_width=True)



def display_demographic_analysis(df: pd.DataFrame):
    st.subheader("Decision Patterns by Applicant Profile")
    demographic_options = [col for col in DEMOGRAPHIC_COLUMNS if col in df.columns]

    if not demographic_options:
        st.info("No demographic columns were available for summary analysis.")
        return

    selected_demo = st.selectbox("Analyze decisions by", demographic_options, format_func=labelize)
    summary = demographic_summary(df, selected_demo)

    insight_source = summary.sort_values("Count", ascending=False).head(1)
    if not insight_source.empty:
        top_group = insight_source.iloc[0][selected_demo]
        top_decision = insight_source.iloc[0]["Decision"]
        top_count = int(insight_source.iloc[0]["Count"])
        render_insight_card(
            "Top Segment Insight",
            f"The largest segment in this view is <b>{top_group}</b>, where <b>{top_decision}</b> appears <b>{top_count}</b> times.",
        )

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(rename_display_columns(summary), use_container_width=True, hide_index=True)
    with col2:
        st.plotly_chart(make_demographic_chart(summary, selected_demo), use_container_width=True)



def display_glossary_section():
    st.subheader("Glossary of Features and Outputs")
    display_pd_definition()
    glossary = load_glossary_data()
    st.dataframe(glossary, use_container_width=True, hide_index=True)



def display_single_application_section(scoring_model, reason_model):
    st.subheader("Score a New Applicant")
    st.caption("Advanced external credit-score and inquiry inputs were removed from the form to keep the experience simpler.")
    display_pd_definition()

    with st.form("credit_risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            code_gender = st.selectbox("Applicant Gender", ["M", "F", "XNA"])
            cnt_children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
            family_status = st.selectbox("Family Status", ["Single / not married", "Married", "Civil marriage", "Separated", "Widow", "Unknown"])
            cnt_fam_members = st.number_input("Number of Family Members", min_value=1.0, max_value=20.0, value=1.0)
            education_type = st.selectbox("Highest Education Level", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
            days_birth = st.number_input("Age in Days Before Application", value=-12000, step=100)
            income_type = st.selectbox("Income Source Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student"])
            occupation_type = st.text_input("Occupation", value="Laborers")
            organization_type = st.text_input("Employer Organization Type", value="Business Entity Type 3")
            days_employed = st.number_input("Days Employed Before Application", value=-2000, step=100)
            amt_income_total = st.number_input("Total Annual Income", min_value=0.0, value=180000.0, step=5000.0)

        with col2:
            flag_own_car = st.selectbox("Owns a Car", ["Y", "N"])
            own_car_age = st.number_input("Age of Applicant's Car", min_value=0.0, value=5.0, step=1.0)
            flag_own_realty = st.selectbox("Owns Real Estate", ["Y", "N"])
            housing_type = st.selectbox("Housing Situation", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"])
            days_registration = st.number_input("Days Since Current Address Registration", value=-4000, step=100)
            days_id_publish = st.number_input("Days Since ID Was Updated", value=-2000, step=100)
            contract_type = st.selectbox("Loan Contract Type", ["Cash loans", "Revolving loans"])
            amt_credit = st.number_input("Loan Amount Requested", min_value=0.0, value=500000.0, step=10000.0)
            amt_annuity = st.number_input("Loan Annuity Amount", min_value=0.0, value=25000.0, step=1000.0)
            amt_goods_price = st.number_input("Goods Price", min_value=0.0, value=450000.0, step=10000.0)

        submitted = st.form_submit_button("Score Application")

    if submitted:
        raw_inputs = {
            "CODE_GENDER": code_gender,
            "CNT_CHILDREN": cnt_children,
            "NAME_FAMILY_STATUS": family_status,
            "CNT_FAM_MEMBERS": cnt_fam_members,
            "NAME_EDUCATION_TYPE": education_type,
            "DAYS_BIRTH": days_birth,
            "NAME_INCOME_TYPE": income_type,
            "OCCUPATION_TYPE": occupation_type,
            "ORGANIZATION_TYPE": organization_type,
            "DAYS_EMPLOYED": days_employed,
            "AMT_INCOME_TOTAL": amt_income_total,
            "FLAG_OWN_CAR": flag_own_car,
            "OWN_CAR_AGE": own_car_age,
            "FLAG_OWN_REALTY": flag_own_realty,
            "NAME_HOUSING_TYPE": housing_type,
            "DAYS_REGISTRATION": days_registration,
            "DAYS_ID_PUBLISH": days_id_publish,
            "NAME_CONTRACT_TYPE": contract_type,
            "AMT_CREDIT": amt_credit,
            "AMT_ANNUITY": amt_annuity,
            "AMT_GOODS_PRICE": amt_goods_price,
        }

        application_df = build_input_dataframe(raw_inputs)
        result = score_application(scoring_model, reason_model, application_df)

        st.write("### Scoring Result")
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Probability of Default (PD)", f"{result['pd']:.4f}")
        metric2.metric("Risk Tier", result["risk_tier"])
        metric3.metric("Decision", result["decision"])

        pill_color = DECISION_COLORS.get(result["decision"], ORANGE)
        st.markdown(f'<span class="decision-pill" style="background:{pill_color};">{result["decision"]}</span>', unsafe_allow_html=True)

        st.write("### Reason Codes")
        for reason in result["reasons"]:
            render_insight_card("Driver", reason)

        st.write("### Input Snapshot")
        st.dataframe(rename_display_columns(pd.DataFrame([raw_inputs])), use_container_width=True, hide_index=True)



def main():
    st.set_page_config(page_title="Credit Risk App", layout="wide")
    inject_theme()
    st.title(APP_TITLE)
    st.write("An interactive credit risk scoring interface with applicant scoring, test-set prediction browsing, and analytics for decisions, tiers, and reason codes.")

    if not SCORING_MODEL_PATH.exists() or not REASON_MODEL_PATH.exists():
        st.error("Required Phase 6 model files were not found. Run src/decisioning.py first to create: phase6_calibrated_scoring_model.joblib and phase6_reason_code_logreg.joblib")
        st.stop()

    scoring_model, reason_model = load_models()
    pred_df = load_prediction_data()
    app_df = load_application_test_data()
    merged_df = merge_predictions_with_inputs(pred_df, app_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio Dashboard",
        "Prediction Browser",
        "Demographic Analytics",
        "Score New Applicant",
        "Glossary"
    ])

    with tab1:
        if merged_df is not None:
            display_overview_section(merged_df)
            st.divider()
            display_reason_analysis(merged_df)
        else:
            st.info("Prediction report not found yet. Run src/decisioning.py to generate reports/phase6_application_test_predictions.csv.")

    with tab2:
        if merged_df is not None:
            display_prediction_browser(merged_df)
        else:
            st.info("Prediction browser is unavailable until the Phase 6 predictions file exists.")

    with tab3:
        if merged_df is not None:
            display_demographic_analysis(merged_df)
        else:
            st.info("Demographic analytics are unavailable until application_test.csv and predictions are both available.")

    with tab4:
        display_single_application_section(scoring_model, reason_model)

    with tab5:
        display_glossary_section()


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.preprocess import select_core_features, clean_features


def get_decision(pd_score):
    if pd_score < 0.05:
        return "APPROVE"
    elif pd_score <= 0.12:
        return "REVIEW"
    else:
        return "DECLINE"


def get_risk_tier(pd_score):
    if pd_score < 0.03:
        return "A"
    elif pd_score < 0.07:
        return "B"
    elif pd_score < 0.12:
        return "C"
    elif pd_score < 0.20:
        return "D"
    else:
        return "E"


def clean_reason_name(feature_name):
    if "__" in feature_name:
        return feature_name.split("__", 1)[1]
    return feature_name


def get_reason_codes(model, x_row, feature_names, top_n=5):
    if not hasattr(model, "feature_importances_"):
        return ["Reason codes unavailable for this model type"]

    importances = model.feature_importances_
    contributions = np.abs(x_row) * importances
    top_idx = np.argsort(contributions)[::-1][:top_n]

    reasons = []
    for i in top_idx:
        feature = clean_reason_name(feature_names[i])
        if x_row[i] > 0:
            reasons.append(f"{feature} contributed to this risk assessment")
        else:
            reasons.append(f"{feature} was an influential factor")

    return reasons


def score_applicant(pipeline, x_raw):
    pd_score = pipeline.predict_proba(x_raw)[0][1]

    decision = get_decision(pd_score)
    tier = get_risk_tier(pd_score)

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    x_transformed = preprocess.transform(x_raw)

    if hasattr(x_transformed, "toarray"):
        x_transformed = x_transformed.toarray()

    x_row = x_transformed[0]
    feature_names = preprocess.get_feature_names_out()
    reasons = get_reason_codes(model, x_row, feature_names, top_n=5)

    return {
        "pd": round(pd_score, 4),
        "decision": decision,
        "risk_tier": tier,
        "reason_codes": reasons
    }


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    pipeline_path = BASE_DIR / "models" / "random_forest_pipeline.joblib"
    test_data_path = BASE_DIR / "data" / "home-credit-default-risk" / "application_test.csv"

    pipeline = joblib.load(pipeline_path)

    df_test = pd.read_csv(test_data_path)

    # apply same preprocessing helper steps used in training
    df_test = select_core_features(df_test)
    df_test = clean_features(df_test)

    for i in range(3):
        sample_applicant = df_test.iloc[[i]].copy()

        if "SK_ID_CURR" in sample_applicant.columns:
            applicant_id = sample_applicant.iloc[0]["SK_ID_CURR"]
            sample_applicant = sample_applicant.drop(columns=["SK_ID_CURR"])
        else:
            applicant_id = None

        result = score_applicant(pipeline, sample_applicant)

        print("\n=== PHASE 6 SCORE REPORT ===")
        if applicant_id is not None:
            print(f"Applicant ID: {applicant_id}")
        print(f"Probability of Default (PD): {result['pd']}")
        print(f"Decision: {result['decision']}")
        print(f"Risk Tier: {result['risk_tier']}")
        print("Reason Codes:")
        for j, reason in enumerate(result["reason_codes"], start=1):
            print(f"{j}. {reason}")

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.preprocess import CORE_FEATURES, clean_features, build_preprocessor


APPROVE_THRESHOLD = 0.05
REVIEW_THRESHOLD = 0.12

# Risk tiers can be adjusted later if your team wants different bands.
RISK_TIERS = [
    (0.00, 0.03, "A"),
    (0.03, 0.06, "B"),
    (0.06, 0.12, "C"),
    (0.12, 0.20, "D"),
    (0.20, 1.01, "E"),
]

REASON_CODE_LABELS = {
    "LOAN_TO_INCOME": "High loan-to-income ratio",
    "DTI_PROXY": "High debt-to-income proxy",
    "AMT_CREDIT": "High requested credit amount",
    "AMT_ANNUITY": "High annuity burden",
    "AMT_REQ_CREDIT_BUREAU_MON": "Many recent credit bureau inquiries",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Many recent credit bureau inquiries",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Many credit bureau inquiries in past year",
    "EXT_SOURCE_1": "Weak external credit score",
    "EXT_SOURCE_2": "Weak external credit score",
    "EXT_SOURCE_3": "Weak external credit score",
    "DAYS_EMPLOYED": "Short employment history",
    "EMPLOYMENT_YEARS": "Short employment history",
    "AMT_INCOME_TOTAL": "Lower income level",
    "AGE": "Younger applicant age",
    "OWN_CAR_AGE": "Older vehicle / weaker asset profile",
    "CNT_CHILDREN": "Higher number of dependents",
}


def select_core_features_for_scoring(df: pd.DataFrame, include_target: bool = False) -> pd.DataFrame:
    """
    In training data, TARGET exists. In application_test.csv it does not.
    This helper keeps the phase 2 feature set while allowing inference data to pass through.
    """
    required = [col for col in CORE_FEATURES if include_target or col != "TARGET"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scoring: {missing}")
    return df[required].copy()


def load_training_data(filepath: str = "../data/home-credit-default-risk/application_train.csv"):
    df = pd.read_csv(filepath)
    df = select_core_features_for_scoring(df, include_target=True)
    df = clean_features(df)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def load_scoring_data(filepath: str = "../data/home-credit-default-risk/application_test.csv"):
    df = pd.read_csv(filepath)

    applicant_ids = None
    if "SK_ID_CURR" in df.columns:
        applicant_ids = df["SK_ID_CURR"].copy()

    X = select_core_features_for_scoring(df, include_target=False)
    X = clean_features(X)

    if applicant_ids is None:
        applicant_ids = pd.Series(np.arange(len(X)), name="SK_ID_CURR")

    return applicant_ids, X


def build_reason_code_pipeline():
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])


def build_calibrated_scoring_model():
    base_pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    return CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5
    )


def get_decision(pd_score: float) -> str:
    if pd_score < APPROVE_THRESHOLD:
        return "APPROVE"
    elif pd_score <= REVIEW_THRESHOLD:
        return "REVIEW"
    return "DECLINE"


def get_risk_tier(pd_score: float) -> str:
    for low, high, tier in RISK_TIERS:
        if low <= pd_score < high:
            return tier
    return "E"


def _safe_feature_name(feature_name: str) -> str:
    # OneHotEncoder uses names like cat__NAME_EDUCATION_TYPE_Higher education
    # We collapse categorical expansions back to the base feature so reason codes stay human-readable.
    if "__" in feature_name:
        feature_name = feature_name.split("__", 1)[1]

    categorical_prefixes = [
        "CODE_GENDER_", "NAME_FAMILY_STATUS_", "NAME_EDUCATION_TYPE_",
        "NAME_INCOME_TYPE_", "OCCUPATION_TYPE_", "ORGANIZATION_TYPE_",
        "FLAG_OWN_CAR_", "FLAG_OWN_REALTY_", "NAME_HOUSING_TYPE_",
        "NAME_CONTRACT_TYPE_"
    ]
    for prefix in categorical_prefixes:
        if feature_name.startswith(prefix):
            return prefix.rstrip("_")
    return feature_name


def _reason_label(base_feature: str, contribution: float) -> str:
    if base_feature in REASON_CODE_LABELS:
        return REASON_CODE_LABELS[base_feature]

    generic_map = {
        "CODE_GENDER": "Applicant demographic category influenced score",
        "NAME_FAMILY_STATUS": "Family status influenced score",
        "NAME_EDUCATION_TYPE": "Education level influenced score",
        "NAME_INCOME_TYPE": "Income type influenced score",
        "OCCUPATION_TYPE": "Occupation category influenced score",
        "ORGANIZATION_TYPE": "Organization type influenced score",
        "FLAG_OWN_CAR": "Vehicle ownership profile influenced score",
        "FLAG_OWN_REALTY": "Real estate ownership profile influenced score",
        "NAME_HOUSING_TYPE": "Housing type influenced score",
        "NAME_CONTRACT_TYPE": "Contract type influenced score",
    }
    if base_feature in generic_map:
        return generic_map[base_feature]

    direction = "higher" if contribution > 0 else "lower"
    return f"{base_feature} contributed {direction} risk"


def generate_reason_codes(reason_pipeline: Pipeline, X_sample: pd.DataFrame, top_n: int = 3):
    preprocess = reason_pipeline.named_steps["preprocess"]
    model = reason_pipeline.named_steps["model"]

    X_transformed = preprocess.transform(X_sample)
    feature_names = preprocess.get_feature_names_out()
    coefficients = model.coef_[0]

    if hasattr(X_transformed, "toarray"):
        row_values = X_transformed.toarray()[0]
    else:
        row_values = np.asarray(X_transformed)[0]

    contributions = row_values * coefficients

    ranked_idx = np.argsort(np.abs(contributions))[::-1]

    reasons = []
    seen = set()

    for idx in ranked_idx:
        if row_values[idx] == 0:
            continue

        base_feature = _safe_feature_name(feature_names[idx])
        if base_feature in seen:
            continue

        label = _reason_label(base_feature, contributions[idx])
        reasons.append(label)
        seen.add(base_feature)

        if len(reasons) == top_n:
            break

    if not reasons:
        reasons = ["Limited signal available from selected features"]

    return reasons


def score_applicant(scoring_model, reason_pipeline, sample: pd.DataFrame):
    pd_score = float(scoring_model.predict_proba(sample)[0, 1])

    return {
        "pd": round(pd_score, 6),
        "decision": get_decision(pd_score),
        "risk_tier": get_risk_tier(pd_score),
        "reason_codes": generate_reason_codes(reason_pipeline, sample, top_n=3)
    }

def score_dataset(scoring_model, reason_pipeline, applicant_ids, X_test):
    print("Scoring entire dataset at once...")

    pd_scores = scoring_model.predict_proba(X_test)[:, 1]
    decisions = [get_decision(pd) for pd in pd_scores]
    risk_tiers = [get_risk_tier(pd) for pd in pd_scores]

    results = []

    for i in range(len(X_test)):
        sample = X_test.iloc[[i]].copy()
        reasons = generate_reason_codes(reason_pipeline, sample, top_n=3)

        results.append({
            "Applicant_ID": applicant_ids.iloc[i] if hasattr(applicant_ids, "iloc") else applicant_ids[i],
            "PD": pd_scores[i],
            "Decision": decisions[i],
            "Risk_Tier": risk_tiers[i],
            "Reason_1": reasons[0] if len(reasons) > 0 else None,
            "Reason_2": reasons[1] if len(reasons) > 1 else None,
            "Reason_3": reasons[2] if len(reasons) > 2 else None,
        })
    import pandas as pd

    return pd.DataFrame(results)

def main():
    os.makedirs("../models", exist_ok=True)
    os.makedirs("../reports", exist_ok=True)

    print("Loading training data...")
    X_train, X_val, y_train, y_val = load_training_data()

    print("Training calibrated Random Forest for PD scoring...")
    scoring_model = build_calibrated_scoring_model()
    scoring_model.fit(X_train, y_train)

    print("Training Logistic Regression for explainable reason codes...")
    reason_pipeline = build_reason_code_pipeline()
    reason_pipeline.fit(X_train, y_train)

    print("Saving trained phase 6 artifacts...")
    joblib.dump(scoring_model, "../models/phase6_calibrated_scoring_model.joblib")
    joblib.dump(reason_pipeline, "../models/phase6_reason_code_logreg.joblib")

    print("Loading application_test.csv for scoring...")
    applicant_ids, X_test = load_scoring_data("../data/home-credit-default-risk/application_test.csv")

    predictions_df = score_dataset(scoring_model, reason_pipeline, applicant_ids, X_test)
    predictions_df.to_csv("../reports/phase6_application_test_predictions.csv", index=False)

    sample_report = predictions_df.head(5).copy()
    sample_report.to_csv("../reports/phase6_sample_score_report.csv", index=False)

    print("\n=== PHASE 6 SAMPLE SCORE REPORT ===")
    for _, row in sample_report.iterrows():
        print("\n----------------------------------------")
        print(f"Applicant ID: {row['Applicant_ID']}") #changed from SK_ID_CURR for debug
        print(f"PD: {row['PD']}")
        print(f"Decision: {row['Decision']}")
        print(f"Risk Tier: {row['Risk_Tier']}")
        print("Reason Codes:")
        for col in ["Reason_1", "Reason_2", "Reason_3"]:
            if col in row and pd.notna(row[col]):
                print(f" - {row[col]}")

    print("\nSaved files:")
    print("- ../models/phase6_calibrated_scoring_model.joblib")
    print("- ../models/phase6_reason_code_logreg.joblib")
    print("- ../reports/phase6_application_test_predictions.csv")
    print("- ../reports/phase6_sample_score_report.csv")


if __name__ == "__main__":
    main()

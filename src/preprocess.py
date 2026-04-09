"""
Feature definitions and preprocessing for Home Credit–style application tables.

Training and inference share the same CORE_FEATURES and clean_features() so that
scores and reason codes stay aligned with the fitted sklearn pipelines.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Columns expected in raw train/test CSVs (TARGET only on train).
CORE_FEATURES = [
    "CODE_GENDER", "CNT_CHILDREN", "NAME_FAMILY_STATUS",
    "CNT_FAM_MEMBERS", "NAME_EDUCATION_TYPE", "DAYS_BIRTH",
    "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "DAYS_EMPLOYED", "AMT_INCOME_TOTAL",
    "FLAG_OWN_CAR", "OWN_CAR_AGE", "FLAG_OWN_REALTY",
    "NAME_HOUSING_TYPE", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "TARGET"
]

# Engineered + raw numeric fields fed through median imputation (after clean_features).
NUMERIC_FEATURES = [
    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "AMT_INCOME_TOTAL",
    "OWN_CAR_AGE", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "AGE", "EMPLOYMENT_YEARS", "DTI_PROXY", "LOAN_TO_INCOME"
]

CATEGORICAL_FEATURES = [
    "CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_HOUSING_TYPE",
    "NAME_CONTRACT_TYPE"
]


def select_core_features(df):
    """Return only CORE_FEATURES (caller must ensure columns exist)."""
    return df[CORE_FEATURES].copy()


def clean_features(df):
    """
    Domain cleaning and feature engineering on a copy of the frame.

    - AGE / EMPLOYMENT_YEARS: from DAYS_* (Kaggle uses negative day counts).
    - 365243 in DAYS_EMPLOYED is a sentinel for “unknown” → treated as NaN.
    - DTI_PROXY / LOAN_TO_INCOME: ratios; divide-by-zero yields inf and is handled
      downstream by imputation inside the sklearn pipeline.
    """
    df = df.copy()

    # Age in years (DAYS_BIRTH is negative days since birth).
    df["AGE"] = -df["DAYS_BIRTH"] / 365.0

    # Unemployed / unknown employment uses 365243 in the raw data.
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365.0

    # Repayment burden and affordability (explicit np.errstate avoids numpy warnings in logs).
    with np.errstate(divide="ignore", invalid="ignore"):
        df["DTI_PROXY"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["LOAN_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    df = df.drop(columns=["DAYS_BIRTH"])

    return df


def build_preprocessor():
    """
    sklearn ColumnTransformer: median/mode imputation + one-hot for categoricals.

    handle_unknown='ignore' keeps inference stable when a category never appeared in train.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    return preprocessor

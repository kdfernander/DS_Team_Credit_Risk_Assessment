import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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
    return df[CORE_FEATURES].copy()


def clean_features(df):
    df = df.copy()

    df["AGE"] = -df["DAYS_BIRTH"] / 365
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYED"] / 365

    df["DTI_PROXY"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["LOAN_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    df = df.drop(columns=["DAYS_BIRTH"])

    return df


def build_preprocessor():
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
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)

from src.preprocess import (
    select_core_features,
    clean_features,
    build_preprocessor
)


def load_data(filepath="../data/home-credit-default-risk/application_train.csv"):
    df = pd.read_csv(filepath)
    df = select_core_features(df)
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


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=42,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )
    }


def train_and_evaluate_models(X_train, X_val, y_train, y_val):
    results = []
    trained_pipelines = {}

    for name, model in get_models().items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]

        train_accuracy = pipeline.score(X_train, y_train)
        val_accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob)
        cm = confusion_matrix(y_val, y_pred)

        trained_pipelines[name] = pipeline

        results.append({
            "Model": name,
            "Train Accuracy": train_accuracy,
            "Validation Accuracy": val_accuracy,
            "Precision": precision,
            "Recall": recall,
            "ROC-AUC": auc
        })

        print(f"\n{'=' * 50}")
        print(name)
        print(f"{'=' * 50}")
        print("Train Accuracy:", train_accuracy)
        print("Validation Accuracy:", val_accuracy)
        print("ROC-AUC Score:", auc)
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(cm)

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    return trained_pipelines, results_df


def choose_best_model(results_df):
    best_model_name = results_df.iloc[0]["Model"]
    return best_model_name


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()

    trained_models, results_df = train_and_evaluate_models(
        X_train, X_val, y_train, y_val
    )

    print("\nFinal Model Comparison Table:")
    print(results_df)

    best_model = choose_best_model(results_df)
    print(f"\nChosen Final Model: {best_model}")
# Credit risk assessment

Small project built around the Home Credit Default Risk data. It fits a probability of default (PD) model, writes scored files, and serves a simple Streamlit app for exploration.

## Install

### With pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### With conda

```bash
conda env create -f environment.yml
conda activate credit-risk-env
```

## How it works

- `src/preprocess.py` – shared feature list and sklearn preprocessor.
- `src/train_phase6.py` – trains calibrated Random Forest + logistic regression for reason codes and saves:
  - `models/phase6_calibrated_scoring_model.joblib`
  - `models/phase6_reason_code_logreg.joblib`
  - `reports/phase6_application_test_predictions.csv`
  - `reports/phase6_sample_score_report.csv`
- `src/streamlit_app.py` – Streamlit dashboard on top of those outputs.
- `src/train_calibration.py` – optional calibration study for a baseline RF model.
- `src/compare_baselines.py` – quick comparison of a few sklearn models.

Data lives under:

- `data/home-credit-default-risk/` – original Kaggle CSVs (train, test, bureau, etc.).
- `data/data_dictionary.csv` – short dictionary used by the app glossary.

## Typical workflow

From the repository root:

```bash
# 1. Train models and generate predictions
python src/train_phase6.py

# 2. Run the app
streamlit run src/streamlit_app.py
```

The app expects:

- the Phase 6 model files in `models/`
- `data/home-credit-default-risk/application_test.csv`
- `reports/phase6_application_test_predictions.csv`

## Layout

```text
.
├── data/
│   ├── data_dictionary.csv
│   └── home-credit-default-risk/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── preprocess.py
│   ├── train_phase6.py
│   ├── streamlit_app.py
│   ├── train_calibration.py
│   └── compare_baselines.py
└── LICENSE
```

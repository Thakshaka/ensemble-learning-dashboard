# Ensemble Learning Dashboard – SL Telecom Churn

MLPR hands-on project.

Predicts whether a telecom customer will churn using bagging, boosting and stacking. Main comparison is Random Forest vs XGBoost on a Sri Lankan-style dataset (Dialog / Mobitel / Hutch / Airtel, districts, LKR bills).

Real operator data isn't public, so the csv in `data/` is synthetic but built around local context.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# mac: xgboost needs OpenMP
# brew install libomp

python make_dataset.py   # only if you want to regenerate the csv
python train_models.py
streamlit run app.py
```

Notebook walkthrough: `ensemble_churn_project.ipynb`

## What's inside

- `app.py` – streamlit dashboard (overview, eda, metrics, try a prediction)
- `train_models.py` – trains RF / XGBoost / stacking and saves stuff under `artifacts/`
- `make_dataset.py` – builds `data/sl_telecom_churn.csv`
- `ensemble_churn_project.ipynb` – full analysis

## Models

- Bagging: Random Forest (+ bagged trees)
- Boosting: Gradient Boosting, XGBoost
- Stacking: RF + GB → logistic regression
- Soft voting as another baseline

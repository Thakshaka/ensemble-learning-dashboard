# fit models and save artifacts for the streamlit app
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "sl_telecom_churn.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

CAT_COLS = ["operator", "district", "plan_type", "last_complaint", "payment_method"]
NUM_COLS = [
    "tenure_months",
    "monthly_bill_lkr",
    "data_usage_gb",
    "call_minutes",
    "late_payments_6m",
    "avg_download_mbps",
    "active_nights_last_month",
    "age",
    "urban",
]


def load_xy():
    df = pd.read_csv(DATA)
    X = df[CAT_COLS + NUM_COLS].copy()
    y = df["churned"].astype(int)
    return df, X, y


def make_preproc():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", NUM_COLS),
        ]
    )


def get_xgb():
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except Exception as e:
        print("xgboost import failed, falling back to HistGradientBoosting:", e)
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=42)


def score_model(name, model, X_test, y_test):
    pred = model.predict(X_test)
    # some estimators might not have predict_proba in edge cases
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
    else:
        proba = None
        auc = None

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, zero_division=0),
    }


def main():
    if not DATA.exists():
        from make_dataset import build

        DATA.parent.mkdir(parents=True, exist_ok=True)
        build().to_csv(DATA, index=False)
        print("dataset created")

    df, X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = make_preproc()

    rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=8, random_state=42),
        n_estimators=40,
        random_state=42,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42
    )
    xgb = get_xgb()

    # stacking: RF + GB -> logistic meta
    stack = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False,
        n_jobs=-1,
    )

    vote = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)),
            ("lr", LogisticRegression(max_iter=1000)),
        ],
        voting="soft",
        n_jobs=-1,
    )

    models = {
        "RandomForest (bagging)": rf,
        "Bagging(DT)": bag,
        "GradientBoosting": gb,
        "XGBoost": xgb,
        "Stacking": stack,
        "Voting (soft)": vote,
    }

    metrics = []
    fitted = {}

    for name, clf in models.items():
        pipe = Pipeline([("pre", make_preproc()), ("clf", clf)])
        print(f"fitting {name} ...")
        pipe.fit(X_train, y_train)
        m = score_model(name, pipe, X_test, y_test)
        # quick 3-fold cv on train (accuracy) – bit slow for stacking but ok
        if name in {"RandomForest (bagging)", "XGBoost", "GradientBoosting"}:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
            m["cv_f1_mean"] = float(cv_scores.mean())
            m["cv_f1_std"] = float(cv_scores.std())
        metrics.append(m)
        fitted[name] = pipe
        print(f"  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']}")

    # feature importance from RF (after one-hot)
    rf_pipe = fitted["RandomForest (bagging)"]
    ohe = rf_pipe.named_steps["pre"].named_transformers_["cat"]
    feat_names = list(ohe.get_feature_names_out(CAT_COLS)) + NUM_COLS
    importances = rf_pipe.named_steps["clf"].feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    feat_imp = [
        {"feature": feat_names[i], "importance": float(importances[i])} for i in top_idx
    ]

    # save stuff
    joblib.dump(fitted["RandomForest (bagging)"], ART / "rf_pipeline.joblib")
    joblib.dump(fitted["XGBoost"], ART / "xgb_pipeline.joblib")
    joblib.dump(fitted["Stacking"], ART / "stack_pipeline.joblib")

    summary = {
        "n_rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "test_size": 0.2,
        "metrics": metrics,
        "feature_importance_rf": feat_imp,
        "columns": CAT_COLS + NUM_COLS,
        "cat_cols": CAT_COLS,
        "num_cols": NUM_COLS,
    }
    (ART / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\ndone. artifacts in", ART)


if __name__ == "__main__":
    main()

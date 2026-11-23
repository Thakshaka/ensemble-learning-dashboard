from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
DATA = ROOT / "data" / "sl_telecom_churn.csv"

st.set_page_config(page_title="SL Telecom Churn", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


@st.cache_data
def load_metrics():
    return json.loads((ART / "metrics.json").read_text())


@st.cache_resource
def load_models():
    return {
        "Random Forest": joblib.load(ART / "rf_pipeline.joblib"),
        "XGBoost": joblib.load(ART / "xgb_pipeline.joblib"),
        "Stacking": joblib.load(ART / "stack_pipeline.joblib"),
    }


def ensure_ready():
    if not DATA.exists() or not (ART / "metrics.json").exists():
        st.warning("Models / data missing – training now (first run only)...")
        import make_dataset
        import train_models

        DATA.parent.mkdir(parents=True, exist_ok=True)
        make_dataset.build().to_csv(DATA, index=False)
        train_models.main()
        st.rerun()


ensure_ready()

df = load_data()
if "last_complaint" in df.columns:
    df["last_complaint"] = df["last_complaint"].fillna("No complaint").astype(str)

meta = load_metrics()
models = load_models()


def choices(series):
    vals = [str(v) for v in series.dropna().unique()]
    return sorted(set(vals))


st.title("Telecom churn – ensemble models")
st.caption("SL operators · RF / XGBoost / stacking")

page = st.sidebar.radio(
    "Section",
    ["Overview", "EDA", "Model comparison", "Try a prediction"],
)

# ---------- Overview ----------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{meta['n_rows']:,}")
    c2.metric("Churn rate", f"{meta['churn_rate']*100:.1f}%")
    c3.metric("Features", len(meta["columns"]))
    best = max(meta["metrics"], key=lambda m: m["f1"])
    c4.metric("Best F1", f"{best['f1']:.3f}", best["model"])

    st.write(
        "Predicting churn for a Sri Lankan-style telecom dataset. "
        "Models: Random Forest (bagging), XGBoost / GB (boosting), stacking and soft voting."
    )
    st.dataframe(df.head(8), use_container_width=True)

# ---------- EDA ----------
elif page == "EDA":
    st.subheader("Quick look at the data")

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        df["churned"].value_counts().sort_index().plot(
            kind="bar", ax=ax, color=["#4c78a8", "#f58518"]
        )
        ax.set_xticklabels(["Stayed (0)", "Churned (1)"], rotation=0)
        ax.set_ylabel("count")
        ax.set_title("Class balance")
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        pd.crosstab(df["operator"], df["churned"], normalize="index").plot(
            kind="bar", ax=ax, stacked=False
        )
        ax.set_ylabel("proportion")
        ax.set_title("Churn share by operator")
        ax.legend(title="churned", labels=["0", "1"])
        st.pyplot(fig)
        plt.close()

    col_c, col_d = st.columns(2)
    with col_c:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for label, g in df.groupby("churned"):
            ax.hist(g["monthly_bill_lkr"], bins=30, alpha=0.55, label=f"churn={label}")
        ax.set_xlabel("monthly bill (LKR)")
        ax.set_title("Bill distribution")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col_d:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for label, g in df.groupby("churned"):
            ax.hist(g["tenure_months"], bins=25, alpha=0.55, label=f"churn={label}")
        ax.set_xlabel("tenure (months)")
        ax.set_title("Tenure vs churn")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.write("Average bill by plan / churn:")
    st.dataframe(
        df.groupby(["plan_type", "churned"])["monthly_bill_lkr"].mean().unstack().round(1),
        use_container_width=True,
    )

# ---------- Model comparison ----------
elif page == "Model comparison":
    st.subheader("Test-set metrics")

    mdf = pd.DataFrame(meta["metrics"])[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    mdf = mdf.sort_values("f1", ascending=False).reset_index(drop=True)
    st.dataframe(
        mdf.style.format(
            {
                "accuracy": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "roc_auc": "{:.3f}",
            }
        ),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(8, 3.8))
    plot_df = mdf.set_index("model")[["accuracy", "f1", "roc_auc"]]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("RF / boosting / stacking side by side")
    ax.legend(loc="lower right")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Confusion matrices (RF vs XGBoost)")
    name_map = {
        "RandomForest (bagging)": "Random Forest",
        "XGBoost": "XGBoost",
    }
    c1, c2 = st.columns(2)
    for col, key in zip([c1, c2], ["RandomForest (bagging)", "XGBoost"]):
        row = next(m for m in meta["metrics"] if m["model"] == key)
        cm = np.array(row["confusion_matrix"])
        with col:
            fig, ax = plt.subplots(figsize=(3.6, 3.2))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["pred 0", "pred 1"])
            ax.set_yticklabels(["actual 0", "actual 1"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            ax.set_title(name_map[key])
            st.pyplot(fig)
            plt.close()

    st.markdown("#### Top features (from Random Forest)")
    fi = pd.DataFrame(meta["feature_importance_rf"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="#4c78a8")
    ax.set_xlabel("importance")
    st.pyplot(fig)
    plt.close()

# ---------- Prediction ----------
elif page == "Try a prediction":
    st.subheader("Score one customer")
    st.write("Change the inputs and compare RF / XGBoost / Stacking probabilities.")

    c1, c2, c3 = st.columns(3)
    with c1:
        operator = st.selectbox("Operator", choices(df.operator))
        district = st.selectbox("District", choices(df.district))
        plan_type = st.selectbox("Plan", choices(df.plan_type))
        payment_method = st.selectbox("Payment", choices(df.payment_method))
        last_complaint = st.selectbox("Last complaint", choices(df.last_complaint))
    with c2:
        tenure_months = st.slider("Tenure (months)", 1, 72, 10)
        monthly_bill_lkr = st.slider("Monthly bill (LKR)", 350, 6500, 2800, step=50)
        data_usage_gb = st.slider("Data usage (GB)", 0.5, 120.0, 25.0)
        call_minutes = st.slider("Call minutes", 20, 1800, 350)
    with c3:
        late_payments_6m = st.slider("Late payments (6m)", 0, 7, 1)
        avg_download_mbps = st.slider("Avg download Mbps", 2.0, 85.0, 18.0)
        active_nights_last_month = st.slider("Active nights last month", 0, 30, 12)
        age = st.slider("Age", 18, 67, 29)
        urban = st.selectbox("Urban area?", [1, 0], format_func=lambda x: "Yes" if x else "No")

    row = pd.DataFrame(
        [
            {
                "operator": operator,
                "district": district,
                "plan_type": plan_type,
                "tenure_months": tenure_months,
                "monthly_bill_lkr": monthly_bill_lkr,
                "data_usage_gb": data_usage_gb,
                "call_minutes": call_minutes,
                "late_payments_6m": late_payments_6m,
                "avg_download_mbps": avg_download_mbps,
                "active_nights_last_month": active_nights_last_month,
                "last_complaint": last_complaint,
                "payment_method": payment_method,
                "age": age,
                "urban": urban,
            }
        ]
    )

    if st.button("Predict churn", type="primary"):
        probs = {}
        preds = {}
        for name, pipe in models.items():
            p = float(pipe.predict_proba(row)[0, 1])
            probs[name] = p
            preds[name] = int(p >= 0.5)

        res = pd.DataFrame(
            {
                "model": list(probs.keys()),
                "P(churn)": list(probs.values()),
                "prediction": ["CHURN" if preds[k] else "stay" for k in probs],
            }
        )
        st.dataframe(res, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.bar(res["model"], res["P(churn)"], color=["#4c78a8", "#f58518", "#54a24b"])
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(churn)")
        st.pyplot(fig)
        plt.close()

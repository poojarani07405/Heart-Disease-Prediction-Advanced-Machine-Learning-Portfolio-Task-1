import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
sns.set_palette(PALETTE)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COL_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


@dataclass
class ModelBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    tuned_models: dict
    results_df: pd.DataFrame
    winner: str
    final_model: object
    feature_list: list
    optimal_threshold: float


@st.cache_data(show_spinner=False)
def load_data():
    df_raw = pd.read_csv(DATA_URL, header=None, names=COL_NAMES, na_values="?")
    df = df_raw.copy()
    df["target"] = (df["num"] > 0).astype(int)
    df.drop(columns="num", inplace=True)
    return df_raw, df


def preprocess(df):
    df_clean = df.copy()
    for col in ["ca", "thal"]:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    for col in ["trestbps", "chol", "thalach", "oldpeak"]:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_clean[col] = df_clean[col].clip(lower, upper)

    df_clean["age_group"] = pd.cut(df_clean["age"], bins=[0, 40, 55, 65, 100], labels=[0, 1, 2, 3]).astype(int)
    df_clean["high_chol"] = (df_clean["chol"] > 200).astype(int)
    df_clean["tachycardia"] = (df_clean["thalach"] > 100).astype(int)
    df_clean["risk_score"] = (
        df_clean["age"] / df_clean["age"].max() * 0.3
        + df_clean["oldpeak"] / max(df_clean["oldpeak"].max(), 1e-9) * 0.3
        + df_clean["ca"] / 3 * 0.4
    )
    return df_clean


@st.cache_resource(show_spinner=False)
def train_models(df):
    df_clean = preprocess(df)
    features = [c for c in df_clean.columns if c != "target"]
    X = df_clean[features]
    y = df_clean["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_baseline = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    # Keep baseline CV results for reference in UI
    baseline_cv = {}
    for name, model in models_baseline.items():
        baseline_cv[name] = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1", "roc_auc", "recall", "precision"],
            return_train_score=True,
        )

    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"], "penalty": ["l2"]},
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0],
        },
        "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
    }
    base_estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    tuned_models = {}
    for name, estimator in base_estimators.items():
        gs = GridSearchCV(estimator, param_grids[name], cv=cv, scoring="f1", n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        tuned_models[name] = gs.best_estimator_

    tuned_models["Decision Tree"] = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
    tuned_models["Decision Tree"].fit(X_train, y_train)

    results = []
    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "AUC-ROC": roc_auc_score(y_test, y_prob),
            }
        )

    results_df = pd.DataFrame(results).set_index("Model").round(3)
    winner = results_df["F1"].idxmax()
    final_model = tuned_models[winner]
    y_prob_final = final_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob_final)
    optimal_threshold = float(thresholds[np.argmax(tpr - fpr)])

    bundle = ModelBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tuned_models=tuned_models,
        results_df=results_df,
        winner=winner,
        final_model=final_model,
        feature_list=features,
        optimal_threshold=optimal_threshold,
    )
    return bundle, baseline_cv


def render_header():
    st.set_page_config(page_title="Heart Disease ML Dashboard", page_icon="🫀", layout="wide")
    st.title("🫀 Heart Disease Prediction Dashboard")
    st.caption("Advanced Machine Learning Portfolio - COM 763 | UCI Cleveland Heart Disease Dataset")
    st.markdown("---")


def plot_missing(df_raw):
    missing = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    missing_df["Missing Count"].plot(kind="bar", ax=axes[0], color="#F44336", edgecolor="black")
    axes[0].set_title("Missing Values by Feature")
    axes[0].set_ylabel("Count")
    for i, (v, pct) in enumerate(zip(missing_df["Missing Count"], missing_df["Missing %"])):
        axes[0].text(i, v + 0.05, f"{v} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    sns.heatmap(df_raw.isnull(), yticklabels=False, cbar=True, cmap="Reds", ax=axes[1])
    axes[1].set_title("Missing Value Heatmap")
    fig.tight_layout()
    return fig


def plot_target(df):
    target_counts = df["target"].value_counts()
    target_labels = ["No Disease (0)", "Disease Present (1)"]
    colors = ["#2196F3", "#F44336"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(target_labels, target_counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Class Distribution (Counts)")
    for i, v in enumerate(target_counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=11, fontweight="bold")

    axes[1].pie(target_counts.values, labels=target_labels, autopct="%1.1f%%", colors=colors, startangle=90)
    axes[1].set_title("Class Proportion")

    sex_target = df.groupby(["sex", "target"]).size().unstack()
    sex_target.index = ["Female", "Male"]
    sex_target.columns = ["No Disease", "Disease"]
    sex_target.plot(kind="bar", ax=axes[2], color=colors, edgecolor="black")
    axes[2].set_title("Disease by Sex")
    axes[2].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def plot_continuous(df):
    continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, feat in enumerate(continuous_features):
        data_no = df[df["target"] == 0][feat].dropna()
        data_yes = df[df["target"] == 1][feat].dropna()
        axes[0, i].hist(data_no, bins=20, alpha=0.6, color="#2196F3", label="No Disease", density=True)
        axes[0, i].hist(data_yes, bins=20, alpha=0.6, color="#F44336", label="Disease", density=True)
        axes[0, i].set_title(feat.upper())
        if i == 0:
            axes[0, i].legend(fontsize=8)

        bp = axes[1, i].boxplot(
            [data_no, data_yes],
            patch_artist=True,
            labels=["No Disease", "Disease"],
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, color in zip(bp["boxes"], ["#2196F3", "#F44336"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    fig.tight_layout()
    return fig


def plot_correlation(df):
    df_corr = df.fillna(df.median(numeric_only=True))
    corr_matrix = df_corr.corr()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Pearson Correlation Matrix")

    target_corr = corr_matrix["target"].drop("target").sort_values()
    colors = ["#F44336" if v > 0 else "#2196F3" for v in target_corr]
    axes[1].barh(target_corr.index, target_corr.values, color=colors, edgecolor="black")
    axes[1].set_title("Feature Correlation with Target")
    axes[1].axvline(0, color="black")
    fig.tight_layout()
    return fig


def plot_model_curves(bundle: ModelBundle):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, model) in enumerate(bundle.tuned_models.items()):
        y_prob = model.predict_proba(bundle.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(bundle.y_test, y_prob)
        axes[0].plot(fpr, tpr, linewidth=2, color=PALETTE[i % len(PALETTE)], label=f"{name} (AUC={auc(fpr, tpr):.3f})")

        prec, rec, _ = precision_recall_curve(bundle.y_test, y_prob)
        axes[1].plot(rec, prec, linewidth=2, color=PALETTE[i % len(PALETTE)], label=name)

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusions(bundle: ModelBundle):
    fig, axes = plt.subplots(1, len(bundle.tuned_models), figsize=(22, 4))
    for ax, (name, model) in zip(axes, bundle.tuned_models.items()):
        y_pred = model.predict(bundle.X_test)
        cm = confusion_matrix(bundle.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        f1 = f1_score(bundle.y_test, y_pred)
        rec = recall_score(bundle.y_test, y_pred)
        ax.set_title(f"{name}\nF1={f1:.2f}, Recall={rec:.2f}", fontsize=9)
    fig.tight_layout()
    return fig


def plot_feature_importance(bundle: ModelBundle):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    targets = [
        ("Random Forest", bundle.tuned_models["Random Forest"]),
        ("Gradient Boosting", bundle.tuned_models["Gradient Boosting"]),
    ]
    for ax, (name, model) in zip(axes, targets):
        fi = pd.Series(model.feature_importances_, index=bundle.feature_list).sort_values(ascending=True)
        ax.barh(fi.index, fi.values, color="#90CAF9", edgecolor="black")
        ax.set_title(f"{name} Feature Importance")
        ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def render_prediction(bundle: ModelBundle):
    st.subheader("Individual Patient Prediction")
    st.caption("Enter patient clinical values and run prediction using selected final model.")

    with st.form("patient_form"):
        c1, c2, c3, c4 = st.columns(4)
        age = c1.slider("Age", 29, 77, 55)
        sex = c2.selectbox("Sex", [0, 1], format_func=lambda x: "Female (0)" if x == 0 else "Male (1)")
        cp = c3.selectbox("Chest Pain Type", [1, 2, 3, 4])
        trestbps = c4.slider("Resting BP", 90, 220, 130)

        c1, c2, c3, c4 = st.columns(4)
        chol = c1.slider("Cholesterol", 100, 600, 240)
        fbs = c2.selectbox("Fasting Blood Sugar >120", [0, 1])
        restecg = c3.selectbox("Rest ECG", [0, 1, 2])
        thalach = c4.slider("Max Heart Rate", 70, 220, 150)

        c1, c2, c3, c4 = st.columns(4)
        exang = c1.selectbox("Exercise Angina", [0, 1])
        oldpeak = c2.slider("Oldpeak", 0.0, 6.5, 1.0, 0.1)
        slope = c3.selectbox("Slope", [1, 2, 3])
        ca = c4.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])

        thal = st.selectbox("Thal", [3, 6, 7])
        use_optimal = st.toggle("Use optimal threshold instead of 0.50", value=True)
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        patient = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                    "thalach": thalach,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal,
                }
            ]
        )
        patient["target"] = 0
        patient = preprocess(patient)
        patient = patient[bundle.feature_list]

        p = bundle.final_model.predict_proba(patient)[0, 1]
        threshold = bundle.optimal_threshold if use_optimal else 0.50
        pred = int(p >= threshold)

        st.metric("Predicted Probability (Disease)", f"{p:.3f}")
        st.metric("Decision Threshold", f"{threshold:.3f}")
        if pred == 1:
            st.error("Prediction: Disease risk detected (Class 1)")
        else:
            st.success("Prediction: No disease detected (Class 0)")


def main():
    render_header()
    df_raw, df = load_data()

    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Go to section",
        [
            "Project Overview",
            "Data & Missing Values",
            "Exploratory Data Analysis",
            "Model Training & Evaluation",
            "Final Model & Prediction",
        ],
    )

    if section == "Project Overview":
        st.subheader("Project Scope")
        st.write("Binary classification to predict heart disease presence using the UCI Cleveland dataset.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_raw.shape[0]}")
        c2.metric("Columns", f"{df_raw.shape[1]}")
        c3.metric("Positive Class Rate", f"{df['target'].mean() * 100:.1f}%")
        st.dataframe(df_raw.head(10), use_container_width=True)

    elif section == "Data & Missing Values":
        st.subheader("Data Schema")
        st.dataframe(df_raw.dtypes.astype(str).rename("dtype"), use_container_width=True)
        st.subheader("Descriptive Statistics")
        st.dataframe(df_raw.describe().T, use_container_width=True)
        st.subheader("Missing Values Analysis")
        st.pyplot(plot_missing(df_raw), clear_figure=True)

    elif section == "Exploratory Data Analysis":
        st.subheader("Target Distribution")
        st.pyplot(plot_target(df), clear_figure=True)
        st.subheader("Continuous Feature Distributions")
        st.pyplot(plot_continuous(df), clear_figure=True)
        st.subheader("Correlation Analysis")
        st.pyplot(plot_correlation(df), clear_figure=True)

    elif section == "Model Training & Evaluation":
        run = st.button("Run Full Training + Hyperparameter Tuning", type="primary")
        if run:
            with st.spinner("Training models and generating evaluation plots..."):
                bundle, baseline_cv = train_models(df)
            st.success("Training complete.")

            st.subheader("Baseline CV Summary (Mean Scores)")
            rows = []
            for name, scores in baseline_cv.items():
                rows.append(
                    {
                        "Model": name,
                        "Accuracy": scores["test_accuracy"].mean(),
                        "Precision": scores["test_precision"].mean(),
                        "Recall": scores["test_recall"].mean(),
                        "F1": scores["test_f1"].mean(),
                        "AUC-ROC": scores["test_roc_auc"].mean(),
                    }
                )
            st.dataframe(pd.DataFrame(rows).set_index("Model").round(3), use_container_width=True)

            st.subheader("Tuned Models - Test Set Metrics")
            st.dataframe(bundle.results_df, use_container_width=True)
            st.subheader("ROC + Precision-Recall Curves")
            st.pyplot(plot_model_curves(bundle), clear_figure=True)
            st.subheader("Confusion Matrices")
            st.pyplot(plot_confusions(bundle), clear_figure=True)

            st.session_state["bundle"] = bundle
        elif "bundle" in st.session_state:
            st.info("Using previously trained models from this session.")
            bundle = st.session_state["bundle"]
            st.dataframe(bundle.results_df, use_container_width=True)
        else:
            st.info("Click the button to run model training and see all evaluation outputs.")

    elif section == "Final Model & Prediction":
        if "bundle" not in st.session_state:
            st.warning("Please run model training first in the 'Model Training & Evaluation' section.")
            return

        bundle = st.session_state["bundle"]
        st.subheader("Final Model Selection")
        st.write(f"Selected model: **{bundle.winner}**")
        st.write(f"Optimal threshold (Youden's J): **{bundle.optimal_threshold:.3f}**")

        st.subheader("Feature Importance (Tree Models)")
        st.pyplot(plot_feature_importance(bundle), clear_figure=True)
        render_prediction(bundle)


if __name__ == "__main__":
    main()

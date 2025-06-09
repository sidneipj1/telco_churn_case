# model_runner.py
# ─────────────────────────────────────────────────────────────
import time, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, log_loss,
    RocCurveDisplay
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
def _safe_display_perf(perf: pd.DataFrame) -> None:
    """Pretty print with .style if Jinja2 is available, else plain text."""
    try:
        import jinja2  # noqa: F401
        display(
            perf.style.format(
                {"auc":"{:.3f}", "accuracy":"{:.3f}", "f1":"{:.3f}",
                 "precision":"{:.3f}", "recall":"{:.3f}", "log_loss":"{:.3f}"}
            )
        )
    except ModuleNotFoundError:
        print(perf.round(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────
def evaluate_models(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    target_col: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train a set of classifiers and return a leaderboard of metrics.
    Also plots ROC curves for the top-3 AUC models.

    Returns
    -------
    pd.DataFrame  –  sorted by AUC (descending).
    """
    # ─── train / test split ──────────────────────────────────────────
    X = df[num_cols + cat_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # ─── preprocessing ──────────────────────────────────────────────
    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # ─── model zoo ─────────────────────────────────────────────────
    models = {
        "LogReg"        : LogisticRegression(max_iter=1000, n_jobs=-1),
        "DecisionTree"  : DecisionTreeClassifier(random_state=random_state),
        "RandomForest"  : RandomForestClassifier(n_estimators=300,
                                                 random_state=random_state,
                                                 n_jobs=-1),
        "GradientBoost" : GradientBoostingClassifier(random_state=random_state),
        "AdaBoost"      : AdaBoostClassifier(random_state=random_state),
        "SVM-RBF"       : SVC(probability=True, random_state=random_state),
        "KNN"           : KNeighborsClassifier(),
        "XGBoost"       : XGBClassifier(
                              n_estimators=400, learning_rate=0.05,
                              max_depth=5, subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=random_state,
                              n_jobs=-1),
        "LightGBM"      : LGBMClassifier(
                              n_estimators=500, learning_rate=0.05,
                              objective="binary", random_state=random_state,
                              n_jobs=-1),
        "CatBoost"      : CatBoostClassifier(
                              iterations=500, learning_rate=0.05,
                              depth=6, verbose=False, random_state=random_state)
    }

    results, yhat_dict = [], {}             # store preds for ROC later

    for name, clf in models.items():
        tic = time.time()

        if name == "CatBoost":
            cat_idx = [X.columns.get_loc(c) for c in cat_cols]
            clf.fit(X_train, y_train, cat_features=cat_idx)
            y_prob = clf.predict_proba(X_test)[:, 1]
        else:
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(X_train, y_train)
            y_prob = pipe.predict_proba(X_test)[:, 1]

        y_pred = (y_prob >= 0.5).astype(int)
        elapsed = round(time.time() - tic, 2)

        yhat_dict[name] = y_prob            # store for ROC

        results.append({
            "model"      : name,
            "auc"        : roc_auc_score(y_test, y_prob),
            "accuracy"   : accuracy_score(y_test, y_pred),
            "f1"         : f1_score(y_test, y_pred),
            "precision"  : precision_score(y_test, y_pred),
            "recall"     : recall_score(y_test, y_pred),
            "log_loss"   : log_loss(y_test, y_prob),
            "train_secs" : elapsed,
        })

    perf = (
        pd.DataFrame(results)
        .sort_values("auc", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== Model leaderboard (test set) ===")
    _safe_display_perf(perf)

    # ─── ROC curves top-3 ──────────────────────────────────────
    top3 = perf.head(3)["model"]
    plt.figure(figsize=(6, 5))
    for name in top3:
        RocCurveDisplay.from_predictions(
            y_test, yhat_dict[name], name=name
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=.4)
    plt.title("ROC curves – top-3 models")
    plt.tight_layout()
    plt.show()

    return perf
from __future__ import annotations
import time, joblib, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, log_loss, RocCurveDisplay,precision_recall_curve)
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
def tune_and_explain(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    target_col: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42,
    param_grid: Dict[str, List] | None = None,
    save_model_path: str | None = None,
    threshold_objective: str = "f1",          # "f1", "precision", or "recall"
    recall_target: float | None = None,       # needed if objective == "recall"
) -> Tuple[pd.DataFrame, Pipeline, float]:
    """
    Fine-tune GradientBoost, pick optimal threshold, plot SHAP, return results.

    Returns
    -------
    metrics_df : DataFrame (one row with AUC, F1, etc. at chosen threshold)
    best_pipe  : fitted Pipeline
    best_thr   : chosen probability cut-off
    """
    if param_grid is None:
        param_grid = {
            "clf__n_estimators": [300, 400, 600],
            "clf__learning_rate": [0.03, 0.05, 0.08],
            "clf__max_depth": [2, 3, 4],
        }

    # ─ split
    X = df[num_cols + cat_cols]
    y = df[target_col].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    # ─ pipeline
    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        verbose_feature_names_out=False)

    pipe = Pipeline([("pre", pre),
                     ("clf", GradientBoostingClassifier(random_state=random_state))])

    # ─ grid search
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc",
                      n_jobs=-1, verbose=0)

    tic = time.time()
    gs.fit(X_tr, y_tr)
    best_pipe  = gs.best_estimator_
    fit_secs   = round(time.time() - tic, 2)

    # ─ probability predictions
    y_prob = best_pipe.predict_proba(X_te)[:, 1]

    # ─ choose threshold via PR curve
    best_thr, pr_metrics = _choose_threshold(
        y_true=y_te,
        y_prob=y_prob,
        objective=threshold_objective,
        recall_target=recall_target,
        plot=True
    )
    y_pred = (y_prob >= best_thr).astype(int)

    # ─ evaluation
    metrics = {
        "AUC"        : roc_auc_score(y_te, y_prob),
        "F1"         : pr_metrics["f1"],
        "Precision"  : pr_metrics["precision"],
        "Recall"     : pr_metrics["recall"],
        "Log-loss"   : log_loss(y_te, y_prob),
        "Threshold"  : round(best_thr, 3),
        "Train_sec"  : fit_secs,
        "Best_params": gs.best_params_
    }
    metrics_df = pd.DataFrame([metrics])
    print("\n=== Test metrics (with tuned threshold) ===")
    print(metrics_df.round(3).to_string(index=False))

    # ─ ROC curve
    RocCurveDisplay.from_predictions(y_te, y_prob)
    plt.title("ROC – tuned GradientBoost"); plt.show()

    # ─ SHAP
    try:
        import shap
        explainer = shap.Explainer(best_pipe["clf"])
        X_te_tr = pd.DataFrame(
            best_pipe["pre"].transform(X_te),
            columns=best_pipe["pre"].get_feature_names_out())
        sv = explainer(X_te_tr, check_additivity=False)
        shap.summary_plot(sv, X_te_tr, plot_type="bar", show=False); plt.show()
        shap.summary_plot(sv, X_te_tr, show=False); plt.show()
    except ImportError:
        print("⚠️  SHAP not installed → pip install shap.")

    # ─ save model
    if save_model_path:
        joblib.dump({"model": best_pipe, "threshold": best_thr}, save_model_path)
        print(f"Model + threshold saved to {save_model_path}")

    return metrics_df, best_pipe, best_thr


# ────────────────────────────────────────────────────────────────────────────
def find_best_threshold(y_true, y_prob, objective="f1", recall_target=None):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1  = 2*prec*rec/(prec+rec+1e-8)

    if objective == "f1":
        idx = np.argmax(f1)
    elif objective == "precision":
        idx = np.argmax(prec)
    elif objective == "recall":
        if recall_target is None:
            raise ValueError("recall_target must be set for objective='recall'")
        cand = np.where(rec >= recall_target)[0]
        idx  = cand[np.argmax(prec[cand])]
    else:
        raise ValueError("objective must be 'f1', 'precision', or 'recall'")

    best_thr = thr[idx]
    metrics  = dict(threshold=round(best_thr,3),
                    precision=round(prec[idx],3),
                    recall   =round(rec[idx],3),
                    f1       =round(f1[idx],3))
    return best_thr, metrics
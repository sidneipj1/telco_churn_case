import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# ──────────────────────────────────────────────────────────
def add_telco_features(
    df: pd.DataFrame,
    *,
    tenure_bins: Tuple[int, ...] = (-np.inf, 12, 24, 60, np.inf),
    tenure_labels: Tuple[str, ...] = ("<12", "12-24", "24-60", "60+"),
    charge_quantiles: Tuple[float, ...] = (0, .25, .50, .75, 1.0),
    charge_labels: Tuple[str, ...] = ("low", "mid-low", "mid-high", "high"),
) -> pd.DataFrame:
    """
    Return a copy of *df* with engineered Telco churn features.
    All original columns are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Telco dataframe.
    tenure_bins, tenure_labels : tuple
        Bin edges & labels for tenure.
    charge_quantiles, charge_labels : tuple
        Quantile edges & labels for monthly charges.

    Returns
    -------
    pd.DataFrame
        Original + engineered columns.
    """
    X = df.copy()

    # ─ Social
    X["has_partner"]     = (X["Partner"] == "Yes").astype(int)
    X["has_dependents"]  = (X["Dependents"] == "Yes").astype(int)
    X["is_senior"]       = X["SeniorCitizen"].astype(int)
    X["social_score"]    = X["has_partner"] + X["has_dependents"] - X["is_senior"]

    # ─ Payment friction
    X["is_electronic_check"] = (X["PaymentMethod"] == "Electronic check").astype(int)
    X["is_automatic"]        = X["PaymentMethod"].str.contains(
        "automatic|credit", case=False, na=False
    ).astype(int)
    X["paperless_billing"]   = (X["PaperlessBilling"] == "Yes").astype(int)

    # ─ Contract
    m_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    X["contract_months"]     = X["Contract"].map(m_map).fillna(1)
    X["is_monthly_contract"] = (X["contract_months"] == 1).astype(int)

    # ─ Products / services
    services = [
        "StreamingTV", "StreamingMovies",
        "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport",
        "PhoneService", "MultipleLines"
    ]
    X["services_count"] = X[services].eq("Yes").sum(axis=1)

    X["has_streaming_pkg"] = (
        X[["StreamingTV", "StreamingMovies"]].eq("Yes").any(axis=1).astype(int)
    )
    online_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    X["has_online_pkg"] = X[online_cols].eq("Yes").any(axis=1).astype(int)

    # ─ Pricing
    X["monthly_charge"] = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
    X["tenure_months"]  = pd.to_numeric(X["tenure"], errors="coerce")

    X["charge_tenure_ratio"] = (
        X["TotalCharges"].replace(" ", np.nan).astype(float)
        / X["tenure_months"].clip(lower=1)
    )
    X["avg_charge_per_service"] = (
        X["monthly_charge"] / X["services_count"].clip(lower=1)
    )

    # ─ Data-quality flag
    X["missing_total_charge"] = (
        X["TotalCharges"].astype(str).str.strip().eq("").astype(int)
    )

    # ─ Binning (categorical for OHE)
    X["tenure_bin"] = pd.cut(
        X["tenure_months"], bins=tenure_bins, labels=tenure_labels
    )
    X["monthly_charge_bin"] = pd.qcut(
        X["monthly_charge"], q=charge_quantiles, labels=charge_labels
    )

    return X

# ──────────────────────────────────────────────────────────
def get_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Return (numeric_cols, categorical_cols) used in the model pipeline.
    Adjust once here if you add/remove engineered variables.
    """
    numeric_cols = [
        "monthly_charge", "tenure_months", "services_count",
        "avg_charge_per_service", "charge_tenure_ratio",
        "social_score",
    ]

    categorical_cols = [
        "contract_months",        # treated as categorical for OHE
        "tenure_bin", "monthly_charge_bin",
        "is_monthly_contract",
        "has_streaming_pkg", "has_online_pkg",
        "is_electronic_check", "is_automatic", "paperless_billing",
        # binary flags = categorical → OHE(drop='first') effectively passes them
    ]
    return numeric_cols, categorical_cols

# ──────────────────────────────────────────────────────────
def build_preprocessor(
    *,
    numeric_cols: List[str] | None = None,
    categorical_cols: List[str] | None = None,
    ohe_drop: str = "first"
) -> ColumnTransformer:
    """
    Assemble a `ColumnTransformer` that one-hot-encodes categorical columns
    and passes numeric columns through unchanged.

    Parameters
    ----------
    numeric_cols : list or None
        If None, uses `get_feature_lists()`.
    categorical_cols : list or None
        Same rule as above.
    ohe_drop : str
        Option forwarded to `OneHotEncoder(drop=...)`.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    if numeric_cols is None or categorical_cols is None:
        num_default, cat_default = get_feature_lists()
        numeric_cols      = numeric_cols or num_default
        categorical_cols  = categorical_cols or cat_default

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("ohe", OneHotEncoder(drop=ohe_drop, handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
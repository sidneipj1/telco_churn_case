import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportions_ztest

# ────────────────────────────────────────────────────────────────────────────
def full_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy table with every column and its dtype.
    """
    return (
        pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
        .reset_index(drop=True)
    )


# ────────────────────────────────────────────────────────────────────────────
def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Columns with NA count > 0, sorted by count.
    If the result is empty, the caller can decide to print a
    'No missing values' message.
    """
    out = (
        df.isna().sum()
          .to_frame("missing")
          .assign(pct=lambda d: d["missing"] / len(df))
          .query("missing > 0")
          .sort_values("missing", ascending=False)
    )
    return out


# ────────────────────────────────────────────────────────────────────────────
def unique_summary(
    df: pd.DataFrame,
    max_show: int = 10,
    numeric_bins: bool = False,
    bin_quantiles: tuple = (0, 0.25, 0.5, 0.75, 1),
) -> pd.DataFrame:
    """
    Summary of *all* columns (numeric & object):
    ─ column
    ─ dtype
    ─ n_unique
    ─ sample_values  (or quantile summary if numeric_bins=True & high cardinality)

    Parameters
    ----------
    max_show : int
        Max number of unique examples to display before adding '…'.
    numeric_bins : bool
        If True, numeric cols with high cardinality get a quantile
        summary instead of listing raw values.
    bin_quantiles : tuple
        Quantiles used when numeric_bins is True.

    Returns
    -------
    pd.DataFrame sorted by n_unique descending.
    """
    rows = []
    for col in df.columns:
        ser = df[col].dropna()
        n_unique = ser.nunique()
        dtype = ser.dtype.name

        if pd.api.types.is_numeric_dtype(ser) and numeric_bins and n_unique > 50:
            qs = ser.quantile(bin_quantiles).round(2).tolist()
            sample = f"quantiles={qs}"
        else:
            uniques = ser.unique()[:max_show]
            sample = ", ".join(map(str, uniques))
            if n_unique > max_show:
                sample += ", …"

        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "n_unique": n_unique,
                "sample_values": sample,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("n_unique", ascending=False)
        .reset_index(drop=True)
    )


# ────────────────────────────────────────────────────────────────────────────
def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Convert TotalCharges; coerce errors -> NaN to fix blanks
    df["TotalCharges"] = (
        df["TotalCharges"]
          .replace(r"^\s*$", np.nan, regex=True)
          .astype("float64")
    )
    
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    
    # SeniorCitizen is 0/1 -> categorical flag
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")

    # mark target as 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
    return df

# ─────────────────────────────────────────────────────────────
def plot_numeric_distribution(df: pd.DataFrame, num_cols: list, bins: int = 30):
    """
    Plot histogram distribution for numeric columns.
    """
    for col in num_cols:
        plt.figure()
        plt.hist(df[col].dropna(), bins=bins, edgecolor="black")
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()


# ─────────────────────────────────────────────────────────────
def plot_numeric_vs_target(df: pd.DataFrame, num_cols: list, target: str = "Churn"):
    """
    Plot boxplots of numeric columns grouped by target.
    """
    for col in num_cols:
        plt.figure()
        df.boxplot(column=col, by=target)
        plt.title(f"{col} by {target}")
        plt.suptitle("")
        plt.ylabel(col)
        plt.show()


# ─────────────────────────────────────────────────────────────
def plot_categorical_vs_target(df: pd.DataFrame, cat_cols: list, target: str = "Churn"):
    """
    Plot mean target (ex: churn rate) per category.
    """
    for col in cat_cols:
        (df.groupby(col)[target]
            .mean()
            .sort_values()
            .plot(kind="bar", rot=0, title=f"P({target}) by {col}", edgecolor="black"))
        plt.ylabel(f"P({target})")
        plt.show()


# ─────────────────────────────────────────────────────────────
def plot_categorical_distribution(df: pd.DataFrame, cat_col: str):
    """
    Plot bar chart of categorical column distribution.
    """
    df[cat_col].value_counts(normalize=True).plot.bar(edgecolor="black")
    plt.title(f"{cat_col} distribution")
    plt.xlabel(cat_col)
    plt.ylabel("Proportion")
    plt.show()


# ─────────────────────────────────────────────────────────────
def plot_bucket_distribution(series: pd.Series, bins: list, labels: list = None, title: str = None):
    """
    Create bins for a numeric series and plot bucket distribution.
    Returns the binned Series.
    """
    binned = pd.cut(series, bins=bins, labels=labels)

    binned.value_counts().sort_index().plot(
        kind="bar", edgecolor="black"
    )
    plt.title(title if title else f"{series.name} distribution by bucket")
    plt.xlabel(series.name + " bin")
    plt.ylabel("Count")
    plt.show()

    return binned


# ─────────────────────────────────────────────────────────────
def plot_churn_rate_by_bin(df, bin_col: str, target_col: str = "Churn"):
    """
    Plot the churn rate (percentage) for each bin or category.
    Display the churn rate value on top of each bar.

    Parameters:
    - df: pandas DataFrame
    - bin_col: column name of the bin or categorical variable 
               (e.g., 'TotalCharges_bin', 'tenure_bin')
    - target_col: binary target variable (default = 'Churn')

    Returns:
    - pandas Series with churn rate by bin
    """
    churn_rate = (
        df.groupby(bin_col)[target_col]
          .mean()
          .sort_index()
    )

    ax = churn_rate.plot(kind="bar", edgecolor="black")

    plt.title(f"{target_col} Rate by {bin_col}")
    plt.xlabel(bin_col)
    plt.ylabel(f"{target_col} Rate")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add labels on top of bars
    for idx, value in enumerate(churn_rate):
        plt.text(
            x=idx,
            y=value + 0.02,  # slightly above the bar
            s=f"{value:.2f}",
            ha="center",
            fontweight="bold"
        )

    plt.show()

    return churn_rate


# ─────────────────────────────────────────────────────────────
def plot_correlation_matrix(df, cols=None, title="Correlation Matrix"):
    """
    Plot correlation matrix with values annotated.
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns

    corr = df[cols].corr()

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()

    ticks = np.arange(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=45, ha="right")
    plt.yticks(ticks, corr.columns)

    # Plot text inside each cell
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                     ha='center', va='center',
                     color="black" if abs(corr.iloc[i, j]) < 0.7 else "white")

    plt.title(title)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
def _cramers_v(table: pd.DataFrame) -> float:
    """
    Cramér’s V effect-size for an RxC contingency table.
    """
    chi2 = chi2_contingency(table, correction=False)[0]
    n    = table.to_numpy().sum()
    r, k = table.shape
    return math.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0


def _effect_label(v: float) -> str:
    """
    Qualitative label for Cramér’s V.
    """
    if   v < 0.10: return "negligible"
    elif v < 0.20: return "weak"
    elif v < 0.30: return "moderate"
    else:          return "strong"


def categorical_target_analysis(
    df: pd.DataFrame,
    target: str = "Churn",
    positive: str | int | bool = "Yes",
    cat_cols: list[str] | None = None,
    p_threshold: float = 0.05,
    v_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Evaluate association between each *categorical* feature and
    a *binary* target using a χ² test (or Fisher for small 2×2 tables).

    Returns a tidy DataFrame with:
    ─ feature            : column name
    ─ levels             : number of unique categories
    ─ p_value            : significance level
    ─ cramers_v          : effect-size (0–1)
    ─ effect_label       : qualitative strength
    ─ churn_rates        : {category: rate} dict rounded to 3 d.p.
    ─ decision           : "keep" if (p < p_threshold and V ≥ v_threshold)

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    target : str
        Name of the binary target column.
    positive : str | int | bool
        Value considered as the positive class (defaults to "Yes").
    cat_cols : list[str] | None
        List of categorical columns to test.  If None, all object columns
        are used.
    p_threshold : float
        Significance cut-off for the hypothesis test.
    v_threshold : float
        Minimum Cramér’s V to consider the effect relevant.

    Returns
    -------
    pd.DataFrame
        Sorted by p-value ascending (most significant first).
    """
    work = df.copy()
    work[target] = (work[target] == positive).astype(int)

    if cat_cols is None:
        cat_cols = work.select_dtypes(include="object").columns.tolist()

    rows = []

    for col in cat_cols:
        if col == target:
            continue

        # contingency table: categories × target (0/1)
        table = pd.crosstab(work[col], work[target])

        # choose χ² or Fisher for 2×2 with small counts
        if table.shape == (2, 2) and (table < 5).any().any():
            _, p_val = fisher_exact(table)
        else:
            _, p_val, _, _ = chi2_contingency(table, correction=False)

        v  = _cramers_v(table)
        cr = (table[1] / table.sum(axis=1)).round(3).to_dict()

        keep = (p_val < p_threshold) and (v >= v_threshold)

        rows.append(
            dict(
                feature      = col,
                levels       = table.shape[0],
                p_value      = p_val,
                cramers_v    = v,
                effect_label = _effect_label(v),
                churn_rates  = cr,
                decision     = "keep" if keep else "drop",
            )
        )

    return (
        pd.DataFrame(rows)
        .sort_values("p_value")
        .reset_index(drop=True)
    )
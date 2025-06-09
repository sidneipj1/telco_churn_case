import pandas as pd

# ────────────────────────────────────────────────────────────────────────────
def scenario_impact(recall: float, precision: float) -> dict:
    caught      = CHURNERS_REAL * recall
    contacted   = caught / precision
    cost        = contacted * COST_PER_CONTACT
    revenue     = caught * SUCCESS_RATE * LTV_12M
    roi         = revenue / cost if cost else float("inf")
    return {
        "Recall"          : round(recall, 3),
        "Precision"       : round(precision, 3),
        "Churners caught" : int(caught),
        "Contacts"        : int(contacted),
        "Cost (R$)"       : round(cost, 0),
        "Revenue saved (R$)": round(revenue, 0),
        "ROI (×)"         : round(roi, 1),
    }

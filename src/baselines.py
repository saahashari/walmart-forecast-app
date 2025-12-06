from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error

def wmae(y_true, y_pred, weights=None):
    if weights is None:
        weights = np.ones_like(y_true, dtype=float)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))

def make_holdout_masks(df_time: pd.DataFrame, holdout_weeks=8, date_col="Date"):
    cutoff = df_time[date_col].max() - pd.Timedelta(weeks=holdout_weeks)
    return cutoff, (df_time[date_col] <= cutoff), (df_time[date_col] > cutoff)

def evaluate_naives(df_sorted: pd.DataFrame, holdout_weeks=8):
    dfx = df_sorted.sort_values(["Store","Dept","Date"])[["Store","Dept","Date","Weekly_Sales"]].copy()
    dfx["lag1"]  = dfx.groupby(["Store","Dept"])["Weekly_Sales"].shift(1)
    dfx["lag52"] = dfx.groupby(["Store","Dept"])["Weekly_Sales"].shift(52)
    cutoff = dfx["Date"].max() - pd.Timedelta(weeks=holdout_weeks)
    train_hist = dfx[dfx["Date"] <= cutoff].copy()
    test = dfx[dfx["Date"] > cutoff].copy()

    weights = (train_hist.groupby(["Store","Dept"])["Weekly_Sales"].mean()
               .rename("w").reset_index())
    test = test.merge(weights, on=["Store","Dept"], how="left")
    test["w"].fillna(test["Weekly_Sales"].mean(), inplace=True)

    out = {}
    for col, name in [("lag1", "Naive(1w)"), ("lag52", "SeasonalNaive(52w)")]:
        te = test.dropna(subset=[col]).copy()
        out[name] = {
            "rows": int(len(te)),
            "MAE": float(mean_absolute_error(te["Weekly_Sales"], te[col])),
            "WMAE": wmae(te["Weekly_Sales"].values, te[col].values, te["w"].values),
        }
    return pd.DataFrame(out).T.reset_index().rename(columns={"index":"Baseline"})

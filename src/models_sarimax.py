from __future__ import annotations

import numpy as np, pandas as pd

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAVE_SM = True
except Exception:
    HAVE_SM = False

def forecast_sarimax_for_series(df: pd.DataFrame, store: int, dept: int, horizon=8):
    if not HAVE_SM:
        return None, "statsmodels not installed"
    s = (df.loc[(df["Store"]==store) & (df["Dept"]==dept), ["Date","Weekly_Sales"]]
           .dropna().sort_values("Date").copy())
    if len(s) < 60:
        return None, "not enough history"
    train = s["Weekly_Sales"].astype(float).values
    try:
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,52), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.forecast(steps=horizon)
        return fc.tolist(), None
    except Exception as e:
        return None, str(e)

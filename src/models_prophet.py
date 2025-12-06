from __future__ import annotations

import numpy as np, pandas as pd

try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

def forecast_prophet_for_series(df: pd.DataFrame, store: int, dept: int, horizon=8):
    if not HAVE_PROPHET:
        return None, "prophet not installed"
    s = (df.loc[(df["Store"]==store) & (df["Dept"]==dept), ["Date","Weekly_Sales"]]
           .dropna().sort_values("Date").copy())
    if len(s) < 20:
        return None, "not enough history"
    ds = s.rename(columns={"Date":"ds","Weekly_Sales":"y"})[["ds","y"]]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(ds)
    future = m.make_future_dataframe(periods=horizon, freq="W")
    fc = m.predict(future).tail(horizon)
    rows = [{"ds": str(r.ds), "yhat": float(r.yhat), "yhat_lower": float(r.yhat_lower), "yhat_upper": float(r.yhat_upper)} for _, r in fc.iterrows()]
    return rows, None

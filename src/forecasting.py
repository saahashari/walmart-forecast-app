from __future__ import annotations
import numpy as np, pandas as pd

def recursive_rf_forecast(series_mod: pd.DataFrame, feature_cols, rf_model, horizon: int = 8) -> tuple[list[str], list[float]]:
    """
    Robust recursive RF forecasting for a single (Store, Dept) slice.

    Parameters
    ----------
    series_mod : DataFrame
        Feature-prepped rows for the target series ONLY, sorted by Date, including:
        - Date, Weekly_Sales
        - lags like Weekly_Sales_lag1/2/52
        - rolling columns like Weekly_Sales_roll4/roll12 (if present)
        - calendar/type/price/markdown features
    feature_cols : list[str]
        The exact feature column order RF expects.
    rf_model : fitted RandomForestRegressor
    horizon : int
        Number of weeks to forecast forward.

    Returns
    -------
    dates : list[str]
    preds : list[float]
    """
    s = series_mod.copy().sort_values("Date").reset_index(drop=True)

    # Keep a small history frame we can append predictions to and recompute derived features.
    hist = s.copy()
    last_date = hist["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")

    preds = []
    for d in future_dates:
        # Start from last row and update date
        row = hist.iloc[[-1]].copy()
        row["Date"] = d

        # Update calendar
        iso = pd.to_datetime(d).isocalendar()
        row["week"], row["month"], row["year"] = int(iso.week), d.month, d.year

        # Update lags using last known/predicted Weekly_Sales
        if "Weekly_Sales" in hist.columns:
            last_y = float(hist.iloc[-1]["Weekly_Sales"])
        else:
            last_y = float(hist.iloc[-1]["Weekly_Sales_lag1"]) if "Weekly_Sales_lag1" in hist.columns else np.nan

        if "Weekly_Sales_lag1" in row.columns:
            row.loc[:, "Weekly_Sales_lag1"] = last_y
        if "Weekly_Sales_lag2" in row.columns:
            prev_lag1 = float(row["Weekly_Sales_lag1"].values[0]) if "Weekly_Sales_lag1" in row.columns else last_y
            row.loc[:, "Weekly_Sales_lag2"] = prev_lag1
        if "Weekly_Sales_lag52" in row.columns and len(hist) >= 52:
            row.loc[:, "Weekly_Sales_lag52"] = float(hist.iloc[-52]["Weekly_Sales"]) if not pd.isna(hist.iloc[-52]["Weekly_Sales"]) else row["Weekly_Sales_lag52"].values[0]

        # Recompute rolling means if present
        for c in [c for c in hist.columns if c.startswith("Weekly_Sales_roll")]:
            try:
                w = int(c.replace("Weekly_Sales_roll", ""))
                recent = hist["Weekly_Sales"].tail(w).values
                if len(recent) > 0 and not np.isnan(recent).all():
                    row.loc[:, c] = float(np.nanmean(recent))
            except Exception:
                pass

        # Align features and predict
        xrow = row.reindex(columns=feature_cols, fill_value=0)
        yhat = float(rf_model.predict(xrow)[0])
        preds.append(yhat)

        # Append predicted point to history so next step can use it
        new_hist_row = row.copy()
        new_hist_row["Weekly_Sales"] = yhat
        hist = pd.concat([hist, new_hist_row], ignore_index=True)

    return future_dates.astype(str).tolist(), preds

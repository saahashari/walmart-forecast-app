from __future__ import annotations
import numpy as np, pandas as pd

def add_group_lag(df: pd.DataFrame, group_cols, target_col: str, lag: int) -> pd.DataFrame:
    df[f"{target_col}_lag{lag}"] = df.groupby(group_cols, as_index=False)[target_col].shift(lag)
    return df

def add_group_roll_mean(df: pd.DataFrame, group_cols, target_col: str, window: int) -> pd.DataFrame:
    df[f"{target_col}_roll{window}"] = (
        df.groupby(group_cols, as_index=False)[target_col]
          .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df

def build_features(
    data: pd.DataFrame,
    lag_list=(1, 2, 52),
    roll_windows=(4, 12),
    include_price=True,
    include_markdowns=True,
    add_interactions=True
):
    dfX = data.sort_values(["Store","Dept","Date"]).copy()

    iso = dfX["Date"].dt.isocalendar()
    dfX["week"]  = iso.week.astype(int)
    dfX["month"] = dfX["Date"].dt.month
    dfX["year"]  = dfX["Date"].dt.year

    for L in lag_list:
        dfX = add_group_lag(dfX, ["Store","Dept"], "Weekly_Sales", L)
    for W in roll_windows:
        dfX = add_group_roll_mean(dfX, ["Store","Dept"], "Weekly_Sales", W)

    if "Type" in dfX.columns:
        dummies = pd.get_dummies(dfX["Type"], prefix="Type", drop_first=False)
        dfX = pd.concat([dfX, dummies], axis=1)

    if "IsHoliday" in dfX.columns:
        dfX["IsHoliday"] = dfX["IsHoliday"].astype(int)

    feat_extra = []
    if include_price:
        for c in ["Fuel_Price","CPI","Unemployment","Temperature"]:
            if c in dfX.columns:
                dfX[c] = pd.to_numeric(dfX[c], errors="coerce")
                feat_extra.append(c)
    if include_markdowns:
        md_cols = [c for c in dfX.columns if c.lower().startswith("markdown")]
        for c in md_cols:
            dfX[c] = pd.to_numeric(dfX[c], errors="coerce").fillna(0.0)
        feat_extra += md_cols

    feature_cols = ["IsHoliday","week","month","year"] +                    [c for c in dfX.columns if c.startswith("Type_")] +                    [c for c in dfX.columns if c.startswith("Weekly_Sales_lag")] +                    [c for c in dfX.columns if c.startswith("Weekly_Sales_roll")] +                    feat_extra

    if add_interactions:
        for c in [c for c in dfX.columns if c.startswith("Type_")]:
            name = f"IsHoliday_x_{c}"
            dfX[name] = dfX["IsHoliday"] * dfX[c]
            feature_cols.append(name)

    if "Weekly_Sales_lag52" in dfX.columns:
        dfX = dfX.dropna(subset=["Weekly_Sales_lag52"])

    X = dfX[feature_cols].copy()
    y = dfX["Weekly_Sales"].copy()
    return dfX, X, y, feature_cols

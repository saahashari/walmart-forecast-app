from __future__ import annotations
import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from .data import load_merge
from .features import build_features
from .baselines import evaluate_naives, make_holdout_masks, wmae
from .models_rf import train_rf, load_model
from .models_sarimax import forecast_sarimax_for_series
from .models_prophet import forecast_prophet_for_series

def leaderboard(data_dir: str|Path, artifacts_dir: str|Path, holdout_weeks=8, topN_series=10, use_trained_rf=True):
    data_dir, artifacts_dir = Path(data_dir), Path(artifacts_dir)
    df = load_merge(data_dir)

    # Baselines across all rows
    base = evaluate_naives(df, holdout_weeks=holdout_weeks)
    rows = []
    for _, r in base.iterrows():
        rows.append({
            "Model": r["Baseline"],
            "Scope": "All rows (test)",
            "Rows": int(r["rows"]),
            "MAE": float(r["MAE"]),
            "WMAE": float(r["WMAE"]),
            "Notes": "Baseline"
        })

    # Build features and split
    mod, X, y, feats = build_features(df)
    cutoff, tr_mask, te_mask = make_holdout_masks(mod, holdout_weeks=holdout_weeks, date_col="Date")
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]

    # RF: load if available, else train quickly
    rf = None
    feat_list = feats
    try:
        from joblib import load
        rf = load(artifacts_dir / "rf_model.joblib")
        feat_list = (artifacts_dir / "rf_features.txt").read_text(encoding="utf-8").splitlines()
        X_te_aligned = X_te.reindex(columns=feat_list, fill_value=0)
    except Exception:
        # Train inline
        from .models_rf import train_rf
        rf, _ = train_rf(X_tr, y_tr, n_splits=5, random_state=42)
        X_te_aligned = X_te

    yhat_te = rf.predict(X_te_aligned)
    w = (mod[tr_mask].groupby(["Store","Dept"])['Weekly_Sales'].mean().rename('w').reset_index())
    w_te = mod.loc[te_mask, ["Store","Dept"]].merge(w, on=["Store","Dept"], how="left")["w"].fillna(y_tr.mean())

    rows.append({
        "Model": "GlobalRF",
        "Scope": "All rows (test)",
        "Rows": int(len(y_te)),
        "MAE": float(mean_absolute_error(y_te, yhat_te)),
        "WMAE": float(wmae(y_te.values, yhat_te, w_te.values)),
        "Notes": "RandomForestRegressor"
    })

    # Classical models on Top-N series (by average sales)
    sstats = (df.groupby(["Store","Dept"])['Weekly_Sales'].mean().rename('avg').reset_index()
                .sort_values('avg', ascending=False).head(topN_series))

    for _, rr in sstats.iterrows():
        store, dept = int(rr.Store), int(rr.Dept)
        s = df.loc[(df["Store"]==store) & (df["Dept"]==dept), ["Date","Weekly_Sales"]].dropna().sort_values("Date")
        if s.empty or len(s) < 60:  # ensure enough history
            continue
        # holdout split
        cut = s["Date"].max() - pd.Timedelta(weeks=holdout_weeks)
        tr = s[s["Date"] <= cut]
        te = s[s["Date"] >  cut]
        if te.empty: continue

        # SARIMAX
        yhat, err = forecast_sarimax_for_series(df, store, dept, horizon=len(te))
        if not err and yhat is not None and len(yhat)==len(te):
            rows.append({
                "Model": "SARIMAX(1,1,1)x(1,1,1,52)",
                "Scope": f"Store {store} Dept {dept}",
                "Rows": int(len(te)),
                "MAE": float(mean_absolute_error(te["Weekly_Sales"].values, np.asarray(yhat))),
                "WMAE": float(wmae(te["Weekly_Sales"].values, np.asarray(yhat), np.full(len(te), float(tr["Weekly_Sales"].mean())))),
                "Notes": "per-series"
            })

        # Prophet
        rows_p = None
        try:
            rows_p, err_p = forecast_prophet_for_series(df, store, dept, horizon=len(te))
        except Exception:
            err_p = "prophet error"
        if rows_p and (not err_p):
            yhat = np.array([r["yhat"] for r in rows_p])
            rows.append({
                "Model": "Prophet",
                "Scope": f"Store {store} Dept {dept}",
                "Rows": int(len(te)),
                "MAE": float(mean_absolute_error(te["Weekly_Sales"].values, yhat)),
                "WMAE": float(wmae(te["Weekly_Sales"].values, yhat, np.full(len(te), float(tr["Weekly_Sales"].mean())))),
                "Notes": "per-series with intervals"
            })

    lb = pd.DataFrame(rows).sort_values(["Scope","WMAE","MAE"]).reset_index(drop=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / "leaderboard.csv"
    lb.to_csv(out, index=False)
    return out, lb

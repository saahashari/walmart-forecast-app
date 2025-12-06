from __future__ import annotations
import argparse, json, numpy as np, pandas as pd
from pathlib import Path
from .data import load_merge
from .features import build_features
from .baselines import make_holdout_masks, wmae, evaluate_naives
from .models_rf import train_rf, save_model, tune_rf

def main(data_dir: str, artifacts_dir: str, holdout_weeks=8, tune=False, n_iter=20, cv_splits=5, random_state=42):
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_merge(data_dir)
    base_df = evaluate_naives(df, holdout_weeks=holdout_weeks)
    base_df.to_csv(artifacts_dir / "baselines.csv", index=False)

    mod, X, y, feature_cols = build_features(df)
    cutoff, tr_mask, te_mask = make_holdout_masks(mod, holdout_weeks=holdout_weeks, date_col="Date")
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    X_te, y_te = X[te_mask], y[te_mask]

    from sklearn.metrics import mean_absolute_error

    if tune:
        rf, best_params = tune_rf(X_tr, y_tr, n_splits=cv_splits, random_state=random_state, n_iter=n_iter)
        (artifacts_dir / "rf_best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    else:
        rf, cv_mae = train_rf(X_tr, y_tr, n_splits=cv_splits, random_state=random_state)

    yhat_te = rf.predict(X_te)

    w = (mod[tr_mask].groupby(["Store","Dept"])['Weekly_Sales'].mean().rename('w').reset_index())
    w_te = mod.loc[te_mask, ["Store","Dept"]].merge(w, on=["Store","Dept"], how="left")["w"].fillna(y_tr.mean())

    scores = {
        "cv_mae": float(np.nan),  # cv_mae computed inside train_rf; optional to store separately
        "holdout_mae": float(mean_absolute_error(y_te, yhat_te)),
        "holdout_wmae": float(wmae(y_te.values, yhat_te, w_te.values))
    }
    pd.DataFrame([scores]).to_csv(artifacts_dir / "rf_scores.csv", index=False)

    from joblib import dump
    dump(rf, artifacts_dir / "rf_model.joblib")
    pd.Series(feature_cols).to_csv(artifacts_dir / "rf_features.txt", index=False, header=False)
    print("Saved model and metrics to", artifacts_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=str)
    ap.add_argument("--artifacts-dir", default="artifacts", type=str)
    ap.add_argument("--holdout-weeks", default=8, type=int)
    ap.add_argument("--tune", action="store_true", help="RandomizedSearchCV on RF with time-aware CV")
    ap.add_argument("--n-iter", default=20, type=int, help="Randomized search iterations")
    ap.add_argument("--cv-splits", default=5, type=int, help="TimeSeriesSplit folds")
    ap.add_argument("--random-state", default=42, type=int)
    args = ap.parse_args()
    main(args.data_dir, args.artifacts_dir, holdout_weeks=args.holdout_weeks, tune=args.tune, n_iter=args.n_iter, cv_splits=args.cv_splits, random_state=args.random_state)

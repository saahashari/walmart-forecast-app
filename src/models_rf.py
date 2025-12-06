from __future__ import annotations

import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump, load

def train_rf(X_tr: pd.DataFrame, y_tr: pd.Series, n_splits=5, random_state=42):
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state
    )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_mae = []
    for tr_idx, va_idx in tscv.split(X_tr):
        rf.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        pred = rf.predict(X_tr.iloc[va_idx])
        cv_mae.append(mean_absolute_error(y_tr.iloc[va_idx], pred))
    rf.fit(X_tr, y_tr)
    return rf, float(np.mean(cv_mae))

def save_model(model, path: str):
    dump(model, path)

def load_model(path: str):
    return load(path)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def tune_rf(X_tr: pd.DataFrame, y_tr: pd.Series, n_splits=5, random_state=42, n_iter=20):
    """Randomized hyperparameter search with time-aware CV."""
    base = RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=random_state)
    param_dist = {
        "n_estimators": randint(300, 900),
        "max_depth": randint(8, 32),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2", None]
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rs = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=random_state, verbose=1
    )
    rs.fit(X_tr, y_tr)
    return rs.best_estimator_, rs.best_params_

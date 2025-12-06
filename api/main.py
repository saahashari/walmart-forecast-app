from __future__ import annotations
import os, json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from joblib import load

from src.data import load_merge
from src.features import build_features
from src.baselines import make_holdout_masks, wmae
from src.forecasting import recursive_rf_forecast
from src.models_sarimax import forecast_sarimax_for_series
from src.models_prophet import forecast_prophet_for_series

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
DATA_DIR = BASE_DIR / "data"
ART_DIR = BASE_DIR / "artifacts"
UI_DIR = BASE_DIR / "ui"

app = FastAPI(title="Walmart Forecast API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DF = None
try:
    DF = load_merge(DATA_DIR)
except Exception as e:
    DF = None

class TrainRequest(BaseModel):
    force: bool = True

class ForecastRequest(BaseModel):
    store: int
    dept: int
    horizon: int = 8
    mode: str = "global_rf"  # global_rf | seasonal_naive | sarimax | prophet
    start_date: str|None = None

@app.get("/", response_class=HTMLResponse)
def root():
    index_html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(index_html)

@app.get("/health")
def health():
    art = [p.name for p in ART_DIR.glob("*.joblib")]
    return {"ok": True, "data_loaded": DF is not None, "artifacts": art}

@app.post("/train")
def train(req: TrainRequest):
    from src.train import main as train_main
    train_main(DATA_DIR, ART_DIR)
    return {"status": "trained"}

def load_rf():
    model_path = ART_DIR / "rf_model.joblib"
    feats_path = ART_DIR / "rf_features.txt"
    if not model_path.exists() or not feats_path.exists():
        return None, None
    rf = load(model_path)
    feature_cols = feats_path.read_text(encoding="utf-8").splitlines()
    return rf, feature_cols


@app.get("/series")
def list_series(top:int=50):
    assert DF is not None, "Data not loaded"
    sstats = (DF.groupby(["Store","Dept"])["Weekly_Sales"].mean().rename("avg").reset_index()
                .sort_values("avg", ascending=False).head(top))
    return {"count": int(len(sstats)), "series": sstats.to_dict(orient="records")}

@app.get("/leaderboard")
def get_leaderboard():
    try:
        from src.evaluate import leaderboard
        path, lb = leaderboard(DATA_DIR, ART_DIR, holdout_weeks=8, topN_series=10)
        return {"path": str(path), "rows": lb.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

@app.post("/forecast")
def forecast(req: ForecastRequest):
    assert DF is not None, "Data not loaded; place CSVs in data/ and restart."
    sub = DF[(DF["Store"]==req.store) & (DF["Dept"]==req.dept)]
    if sub.empty:
        return {"error": f"No data for Store {req.store}, Dept {req.dept}"}

    if req.mode == "seasonal_naive":
        s = sub.sort_values("Date").copy()
        s["lag52"] = s["Weekly_Sales"].shift(52)
        last = s["lag52"].dropna().values
        if len(last) == 0:
            return {"error": "Not enough history for seasonal naive (need >=52 weeks)"}
        yhat = [float(last[-1])] * req.horizon
        dates = pd.date_range(s["Date"].max() + pd.Timedelta(weeks=1), periods=req.horizon, freq="W")
        return {"mode": "seasonal_naive", "dates": dates.astype(str).tolist(), "yhat": yhat}

    if req.mode == "sarimax":
        yhat, err = forecast_sarimax_for_series(DF, req.store, req.dept, horizon=req.horizon)
        if err: return {"error": err}
        dates = pd.date_range(sub["Date"].max() + pd.Timedelta(weeks=1), periods=req.horizon, freq="W")
        return {"mode": "sarimax", "dates": dates.astype(str).tolist(), "yhat": yhat}

    if req.mode == "prophet":
        rows, err = forecast_prophet_for_series(DF, req.store, req.dept, horizon=req.horizon)
        if err: return {"error": err}
        return {"mode": "prophet", "rows": rows}

    rf, feature_cols = load_rf()
    if rf is None:
        return {"error": "RF model not trained. Call /train first."}

    mod, X, y, feats = build_features(DF)
    X = X.reindex(columns=feature_cols, fill_value=0)
    series_mod = mod[(mod["Store"]==req.store) & (mod["Dept"]==req.dept)].copy().sort_values("Date")
    if series_mod.empty:
        return {"error": "Series has no rows after feature prep"}
    future_dates = pd.date_range(series_mod["Date"].max() + pd.Timedelta(weeks=1), periods=req.horizon, freq="W")
    dates, preds = recursive_rf_forecast(series_mod, feature_cols, rf, req.horizon)
    return {"mode": "global_rf", "dates": dates, "yhat": preds}


@app.post("/plot")
def plot(req: ForecastRequest):
    """Return a PNG chart for the requested forecast mode."""
    assert DF is not None, "Data not loaded; place CSVs in data/ and restart."
    sub = DF[(DF["Store"]==req.store) & (DF["Dept"]==req.dept)].copy().sort_values("Date")
    if sub.empty:
        return {"error": f"No data for Store {req.store}, Dept {req.dept}"}

    fig, ax = plt.subplots(figsize=(10, 4))
    # Plot recent history (last ~2 years to avoid clutter)
    hist = sub.tail(104)
    ax.plot(hist["Date"], hist["Weekly_Sales"], label="Actual", linewidth=2)

    # Compute forecast using same logic as /forecast
    if req.mode == "seasonal_naive":
        s = sub.copy()
        s["lag52"] = s["Weekly_Sales"].shift(52)
        last_val = s["lag52"].dropna().values
        if len(last_val) == 0:
            buf = io.BytesIO()
            fig.suptitle("Not enough history for seasonal naive", color="orange")
            fig.tight_layout(); fig.savefig(buf, format="png", dpi=144); plt.close(fig)
            buf.seek(0); return StreamingResponse(buf, media_type="image/png")
        yhat = [float(last_val[-1])] * req.horizon
        dates = pd.date_range(sub["Date"].max() + pd.Timedelta(weeks=1), periods=req.horizon, freq="W")
        ax.plot(dates, yhat, label="Seasonal Naive", linestyle="--")

    elif req.mode == "sarimax":
        yhat, err = forecast_sarimax_for_series(DF, req.store, req.dept, horizon=req.horizon)
        dates = pd.date_range(sub["Date"].max() + pd.Timedelta(weeks=1), periods=req.horizon, freq="W")
        if err or yhat is None:
            fig.suptitle(f"SARIMAX error: {err}", color="orange")
        else:
            ax.plot(dates, yhat, label="SARIMAX", linestyle="--")

    elif req.mode == "prophet":
        rows, err = forecast_prophet_for_series(DF, req.store, req.dept, horizon=req.horizon)
        if err or not rows:
            fig.suptitle(f"Prophet error: {err}", color="orange")
        else:
            ds = [r["ds"] for r in rows]
            yhat = [r["yhat"] for r in rows]
            lo = [r["yhat_lower"] for r in rows]
            hi = [r["yhat_upper"] for r in rows]
            ax.plot(ds, yhat, label="Prophet")
            ax.fill_between(ds, lo, hi, alpha=0.2, label="CI")

    else:  # global_rf
        rf, feature_cols = load_rf()
        if rf is None:
            fig.suptitle("RF not trained; call /train first.", color="orange")
        else:
            mod, X, y, feats = build_features(DF)
            series_mod = mod[(mod["Store"]==req.store) & (mod["Dept"]==req.dept)].copy().sort_values("Date")
            if series_mod.empty:
                fig.suptitle("Series empty after feature prep", color="orange")
            else:
                dates, preds = recursive_rf_forecast(series_mod, feature_cols, rf, req.horizon)
                ax.plot(pd.to_datetime(dates), preds, label="Global RF", linestyle="--")

    ax.set_title(f"Store {req.store} Dept {req.dept} — {req.mode} forecast")
    ax.set_ylabel("Weekly_Sales")
    ax.legend(); fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144)
    plt.close(fig)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

app.mount('/static', StaticFiles(directory=str(UI_DIR)), name='static')

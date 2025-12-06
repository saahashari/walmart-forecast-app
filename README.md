# Walmart Weekly Department Sales Forecast — Full Application

This repo combines:
- **EDA & Feature Selection** (Notebook)  
- **Model Training/Testing/Selection** with strong baselines (Notebook + Python modules)  
- **Tuning & Add-ons** (RF + classical SARIMAX/Prophet options)  
- **Production-ish API** using FastAPI with a simple **UI** to request forecasts

## Quick Start

1. Place Kaggle data in `data/`:
   - `train.csv`, `features.csv`, `stores.csv`

2. (Recommended) Create a virtual env:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Explore the notebook(s) in `notebooks/`:
   - `01_EDA_and_Feature_Selection.ipynb`
   - `02_Model_Training_Selection.ipynb`

4. Train the global RF model from CLI (saves to `artifacts/rf_model.joblib`):
   ```bash
   python -m src.train --data-dir data --artifacts-dir artifacts
   ```

5. Run the API (FastAPI):
   ```bash
   uvicorn api.main:app --reload
   ```

6. Open the UI:
   - Visit `http://localhost:8000` (served by the API's static mount) or open `ui/index.html` and set `API_BASE` in `ui/app.js`.

## Endpoints
- `GET /health` — quick health check
- `POST /train` — trains global RF; body: `{"force": true}` to retrain
- `POST /forecast` — body:
  ```json
  {
    "store": 1,
    "dept": 1,
    "horizon": 8,
    "mode": "global_rf",
    "start_date": null
  }
  ```
  - `mode`: `global_rf` (default), `seasonal_naive`, `sarimax` (if statsmodels present), `prophet` (if prophet installed)

## Notes
- RF: **global** cross-sectional model (lags/rolls/holiday/type/markdowns). Time-aware CV + fixed holdout used in notebook.
- SARIMAX/Prophet: **per-series** classical models; intervals are supported where available.
- Baselines: last-value and seasonal-naive provided.


## Leaderboard (Unified Holdout)
Run a consistent comparison across **Baselines**, **Global RF**, and per-series **SARIMAX/Prophet** (Top-N by volume):
```bash
python -c "from src.evaluate import leaderboard; print(leaderboard('data','artifacts'))"
# or open notebooks/03_Leaderboard.ipynb
```
Outputs `artifacts/leaderboard.csv` with MAE/WMAE and row counts.

## UI Tips
- Use **Global RF** for fast, cross-sectional forecasting.
- Use **Prophet** to get **interval bands** in the chart.
- Click **Load Top Series** to see high-volume `(Store, Dept)` options.
- Click **Run Leaderboard** to generate and view the latest comparison.


## Tuning RF
Run randomized search with time-series CV and save best params:
```bash
python -m src.train --data-dir data --artifacts-dir artifacts --tune --n-iter 30 --cv-splits 5
# writes artifacts/rf_best_params.json
```

## Plot Endpoint (PNG)
Request a forecast chart (PNG):
```bash
curl -X POST http://localhost:8000/plot -H "content-type: application/json"   -d '{"store":1,"dept":1,"horizon":8,"mode":"prophet"}' --output forecast.png
```

from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_merge(data_dir: str|Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")
    on_cols = ["Store", "Date"]
    if "IsHoliday" in train.columns and "IsHoliday" in features.columns:
        on_cols.append("IsHoliday")
    df = (
        train.merge(features, on=on_cols, how="left")
             .merge(stores, on="Store", how="left")
             .sort_values(["Store","Dept","Date"])
             .reset_index(drop=True)
    )
    return df

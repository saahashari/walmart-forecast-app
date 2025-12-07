# scripts/make_sample.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Create a small demo dataset (data/sample/) from full Walmart files.")
    ap.add_argument("--data-dir", default="data", help="Folder with full train.csv, features.csv, stores.csv")
    ap.add_argument("--out-dir", default="data/sample", help="Where to write the sample CSVs")
    ap.add_argument("--weeks", type=int, default=52, help="Approx # of weeks to include")
    ap.add_argument("--stores", type=int, nargs="*", default=None, help="Store IDs to include (optional)")
    ap.add_argument("--depts", type=int, nargs="*", default=None, help="Dept IDs to include (optional)")
    ap.add_argument("--topk", type=int, default=2, help="Auto-pick top-K (Store,Dept) by row count if no stores/depts given")
    return ap.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(data_dir / "stores.csv")

    # Choose which (Store, Dept) series to include
    if args.stores and args.depts:
        pairs = [(s, d) for s in args.stores for d in args.depts]
        keep = pd.DataFrame(pairs, columns=["Store", "Dept"]).astype(int)
    else:
        stats = (train.groupby(["Store", "Dept"])
                       .size().reset_index(name="rows")
                       .sort_values("rows", ascending=False)
                       .head(args.topk))
        keep = stats[["Store", "Dept"]].copy()

    t = train.merge(keep, on=["Store", "Dept"], how="inner")

    # Pick last ~N weeks window
    max_date = t["Date"].max()
    start_date = max_date - pd.Timedelta(days=args.weeks * 7)
    t = t[t["Date"] >= start_date].copy()

    keep_stores = sorted(t["Store"].unique().tolist())
    f = features[(features["Store"].isin(keep_stores)) & (features["Date"] >= start_date)].copy()
    s = stores[stores["Store"].isin(keep_stores)].copy()

    t = t.sort_values(["Store", "Dept", "Date"])
    f = f.sort_values(["Store", "Date"])
    s = s.sort_values(["Store"])

    (out_dir / "train.csv").write_text(t.to_csv(index=False))
    (out_dir / "features.csv").write_text(f.to_csv(index=False))
    (out_dir / "stores.csv").write_text(s.to_csv(index=False))

    print(f"Sample written to: {out_dir}")
    print(f"  train:    {t.shape}")
    print(f"  features: {f.shape}")
    print(f"  stores:   {s.shape}")
    print("Included stores:", keep_stores)
    print("Date range:", t["Date"].min().date(), "→", t["Date"].max().date())

if __name__ == "__main__":
    sys.exit(main())

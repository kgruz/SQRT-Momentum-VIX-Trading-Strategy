import pandas as pd

existing = pd.read_csv("constituents.csv")
missing  = pd.read_csv("missing_sectors.csv")

sym_col = next(c for c in existing.columns if c.lower() in ["symbol", "ticker"])
sec_col = next(c for c in existing.columns if "sector" in c.lower())

missing = missing.rename(columns={"Symbol": sym_col, "Sector": sec_col})

combined = pd.concat([existing[[sym_col, sec_col]], missing[[sym_col, sec_col]]], ignore_index=True)
combined = combined.drop_duplicates(subset=[sym_col], keep="first")
combined.to_csv("constituents_full.csv", index=False)

print(f"Done: {len(combined)} tickers in constituents_full.csv")
print(combined[sec_col].value_counts().to_string())
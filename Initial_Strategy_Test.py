import pandas as pd
import os

vix = pd.read_csv("SQRT-Momentum-VIX-Trading-Strategy/vix_2016_2023.csv", index_col=0, parse_dates=True)
print(vix.head())



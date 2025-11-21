import yfinance as yf

vix = yf.download("^VIX",
                  start="2016-01-01",
                  end="2024-01-01")  # end date is exclusive, so use 2024-01-01

print(vix.head())
print(vix.tail())

vix.to_csv("vix_2016_2023.csv")
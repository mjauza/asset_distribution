from financetoolkit import Toolkit
import os
import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator, ppo, StochasticOscillator
def get_hist_data(
    companies_symbols,
    start_dt_str,
    end_dt_str,
    add_tech_indi = True
):
    FMP_KEY = os.getenv("FMP")
    companies = Toolkit(companies_symbols, api_key=FMP_KEY)
    historical_data = companies.get_historical_data(start=start_dt_str, end=end_dt_str)

    res = {}
    for symbol in companies_symbols:
        if len(companies_symbols) == 1:
            cols = ["Close", "Volume", "High", "Low"]
        else:
            cols = [("Close", symbol), ("Volume", symbol), ("High", symbol), ("Low", symbol)]
        df = historical_data[cols]
        df.columns = ["Close", "Volume", "High", "Low"]
        df.fillna(method="ffill", inplace=True)
        df.index = pd.to_datetime(df.index)
        # add technical indicators
        if add_tech_indi:
            add_ta_indi(df)
        res[symbol] = df

    return res

def add_ta_indi(df):
    # MACD
    macd_obj = MACD(df["Close"])
    macd_line = macd_obj.macd()
    macd_signal = macd_obj.macd_signal()
    macd_diff = (macd_line - macd_signal) > 0
    macd_buy = (macd_diff == True) & (macd_diff.shift(1) == False)
    macd_sell = (macd_diff == False) & (macd_diff.shift(1) == True)
    macd_buy = macd_buy.astype(int)
    macd_sell = -macd_sell.astype(int)
    macd_buy_sell = macd_buy + macd_sell
    df["macd"] = macd_buy_sell

    # RSI
    rsi_ser = RSIIndicator(df["Close"]).rsi()
    rsi_buy = (rsi_ser > 30) & (rsi_ser.shift(1) < 30)
    rsi_buy = rsi_buy.astype(int)
    rsi_sell = (rsi_ser < 70) & (rsi_ser.shift(1) > 70)
    rsi_sell = -rsi_sell.astype(int)
    df["rsi"] = rsi_buy + rsi_sell

    # STochastic Oscilator
    so_ser = StochasticOscillator(df["High"],df["Low"],df["Close"]).stoch()
    so_buy = (so_ser > 20) & (so_ser.shift(1) < 20)
    so_buy = so_buy.astype(int)
    so_sell = (so_ser < 80) & (so_ser.shift(1) > 80)
    so_sell = -so_sell.astype(int)
    df["so"] = so_buy + so_sell

if __name__ == "__main__":
    # get_hist_data(
    #     ["^GSPC"],
    #     "1985-01-01",
    #     "2020-12-31"
    # )

    res = get_hist_data(
        ["AAPL"],
        "1985-01-01",
        "2020-12-31"
    )
    print(res)
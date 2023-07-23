from financetoolkit import Toolkit
import os
import pandas as pd
def get_hist_data(
    companies_symbols,
    start_dt_str,
    end_dt_str
):
    FMP_KEY = os.getenv("FMP")
    companies = Toolkit(companies_symbols, api_key=FMP_KEY)
    historical_data = companies.get_historical_data(start=start_dt_str, end=end_dt_str)

    res = {}
    for symbol in companies_symbols:
        if len(companies_symbols) == 1:
            cols = ["Close", "Volume"]
        else:
            cols = [("Close", symbol), ("Volume", symbol)]
        df = historical_data[cols]
        df.columns = ["Close", "Volume"]
        df.fillna(method="ffill", inplace=True)
        df.index = pd.to_datetime(df.index)
        res[symbol] = df

    return res


if __name__ == "__main__":
    res = get_hist_data(
        ["AAPL"],
        "1985-01-01",
        "2020-12-31"
    )
    print(res)
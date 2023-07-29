from fin_ratios import get_fin_ratios
from hist_data import get_hist_data
import pandas as pd
import os
def convert_ratios_data(ratios_df):
    df_new = ratios_df.transpose()
    df_new["year"] = df_new.index.year
    df_new["quarter"] = df_new.index.quarter
    return df_new
def get_raw_data(
    companies_symbols
):
    # get ratios data
    fin_ratios_data = get_fin_ratios(companies_symbols)
    res = {}
    for symbol in companies_symbols:
        years = fin_ratios_data[symbol].columns.year
        start_dt_str = f"{min(years)}-01-01"
        end_dt_str = f"{max(years)}-12-31"
        symbol_hist_data = get_hist_data(
            [symbol],
            start_dt_str,
            end_dt_str
        )
        symbol_hist_data[symbol]["year"] = symbol_hist_data[symbol].index.year
        symbol_hist_data[symbol]["quarter"] = symbol_hist_data[symbol].index.quarter
        df_ratios = convert_ratios_data(fin_ratios_data[symbol])
        df_merged = symbol_hist_data[symbol].merge(df_ratios, on = ["year", "quarter"])
        df_merged.drop(["year", "quarter"], axis=1, inplace=True)
        res[symbol] = df_merged

    return res

def append_target(raw_data_dict, num_day_ahead = 1, target_var = "Close", get_ret = True):
    res = {}
    for symbol, df in raw_data_dict.items():
        if get_ret:
            df["target"] = (df[target_var].shift(-num_day_ahead) - df[target_var])/df[target_var]
        else:
            df["target"] = df[target_var].shift(-num_day_ahead)
        res[symbol] = df
    return res

def final_touch(res_dict):
    res = {}
    for symbol, df in res_dict.items():
        res[symbol] = df.dropna()
    return res
def data_pipeline(
    companies_symbols,
    num_day_ahead = 1,
    target_var = "Close",
    save_dir="../asset_dist_data"
):
    raw_data_dict = get_raw_data(companies_symbols)
    target_dict = append_target(raw_data_dict, num_day_ahead, target_var)
    final_res = final_touch(target_dict)
    save_to_disk(final_res, save_dir)

def save_to_disk(final_res, save_dir = "../asset_dist_data"):
    for sym, df in final_res.items():
        filename = f"{sym}.parquet"
        fullpath = os.path.join(save_dir, filename)
        df.to_parquet(fullpath)


if __name__ == "__main__":
    data_pipeline(["AAPL"])



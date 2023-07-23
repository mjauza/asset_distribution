import pandas as pd
import os
def load_data(company_symbol, save_dir = "../asset_dist_data"):
    filename = f"{company_symbol}.parquet"
    fullpath = os.path.join(save_dir, filename)
    df = pd.read_parquet(fullpath)
    return df

def get_train_test(df, test_size = 0.20):
    N_test = int(len(df) * test_size)
    df_test = df.iloc[-N_test:, :]
    df_train = df.iloc[:-N_test, :]
    return df_train, df_test




import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
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


def evaluate(y_true: np.array, y_pred: np.array, is_return):
    r2 = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)**0.5
    res = {
        "r2" : r2,
        "rmse": rmse
    }
    if is_return:
        up_true = (y_true > 0).astype(int)
        up_pred = (y_pred > 0).astype(int)
        acc = accuracy_score(up_true, up_pred)
        res["acc"] = acc

    return res

def plot_history(history):
    df = pd.DataFrame({
        "loss" : history.history["loss"],
        "val_loss": history.history["val_loss"],
    })
    df.plot()
    plt.show()
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from model_utils import load_data, get_train_test
import pickle
import os
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

def train_model(X,y, symbol = None, save = False):
    kernel = RBF()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state = 0, n_restarts_optimizer = 5, normalize_y = True)
    gpr.fit(X, y)

    if save:
        save_dir = "../asset_dist_data"
        filename = f"GP_{symbol}.pickle"
        fullpath = os.path.join(save_dir, filename)
        with open(fullpath, 'wb') as handle:
            pickle.dump(gpr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return gpr

def load_model(symbol):
    save_dir = "../asset_dist_data"
    filename = f"GP_{symbol}.pickle"
    fullpath = os.path.join(save_dir, filename)
    with open(fullpath, 'rb') as handle:
        gpr = pickle.load(handle)
    return gpr

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

def pipeline(company_symbol, train_model_bool = False, is_return = True):
    df = load_data(company_symbol)
    df_train, df_test = get_train_test(df)
    del df

    # train model
    if train_model_bool:
        X_train = df_train.drop("target", axis=1).values
        y_train = df_train["target"].values
        gpr = train_model(X_train, y_train, company_symbol)
    else:
        gpr = load_model(company_symbol)
    # test model
    X_test = df_test.drop("target", axis=1).values
    y_test = df_test["target"].values
    y_pred, y_stdev = gpr.predict(X_test, return_std=True)
    eval_res = evaluate(y_test, y_pred, is_return)
    print(eval_res)


if __name__ == "__main__":
    pipeline("AAPL",train_model_bool = True, is_return = True)

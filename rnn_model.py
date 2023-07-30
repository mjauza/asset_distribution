import pandas as pd
import numpy as np
from model_utils import load_data, get_train_test, evaluate, plot_history
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
import tensorflow as tf
tfkl = tf.keras.layers
from sklearn.preprocessing import StandardScaler

def get_model(lag, num_feat, lr = 1e-4):
    reg = tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-4)
    model = tf.keras.Sequential([
        tfkl.GRU(64, activation='relu', input_shape=(lag, num_feat), return_sequences=True),
        tfkl.GRU(32, activation='relu', return_sequences=False),
        tfkl.Dense(2,
                   kernel_initializer='random_normal',
                   bias_initializer='zeros'),
        tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=1e-6 + tf.math.abs(t[..., 1:])))
    ])
    negloglik = lambda y_true, y_pred: -y_pred.log_prob(y_true)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=negloglik)
    return model


def train_model(X,y, lag, batch_size = 256, val_size = 0.05):
    num_feat = X.shape[1]
    model = get_model(lag,num_feat)
    # get train and val
    N_val = int(len(X) * val_size)
    X_train, y_train = X[:-N_val, :], y[:-N_val]
    X_val, y_val = X[-N_val:, :], y[-N_val:]

    # get callback
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=20, min_delta=1e-1, mode='auto',
                                                baseline=None, restore_best_weights=True)

    train_generator = TimeseriesGenerator(X_train,y_train, length=lag, batch_size = batch_size)
    val_generator = TimeseriesGenerator(X_val, y_val, length=lag, batch_size=batch_size)
    history = model.fit_generator(train_generator, epochs=200, verbose=0, validation_data=val_generator, callbacks=[callback])
    plot_history(history)
    return model


def get_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def get_predictions(model, X,y, lag):
    # ds = tf.keras.utils.timeseries_dataset_from_array(
    #     data = X,
    #     targets = y,
    #     sequence_length = lag,
    #     batch_size = None
    # )
    # y_pred = []
    # for x, _ in ds.as_numpy_iterator():
    #     y_pred.append(model.predict(x.reshape((1, lag, -1)))[0,0])
    generator = TimeseriesGenerator(X, y, length=lag, batch_size=len(y))
    y_pred = model.predict(generator)
    y_true = generator[0][1]
    return y_pred.reshape((-1 , )), y_true

def pipeline(company_symbol):
    df = load_data(company_symbol)
    df_train, df_test = get_train_test(df)
    del df

    LAG = 4
    BATCH_SIZE = 256
    # train model
    X_train = df_train.drop("target", axis=1).values
    y_train = df_train["target"].values

    # scale data
    scaler = get_scaler(X_train)
    X_train_scld = scaler.transform(X_train)

    # train model
    model = train_model(X_train_scld, y_train, lag = LAG, batch_size = BATCH_SIZE, val_size=0.05)

    # evaluate model
    X_test = df_test.drop("target", axis=1).values
    y_test = df_test["target"].values
    X_test_scld = scaler.transform(X_test)

    y_pred, y_true = get_predictions(model, X_test_scld, y_test, lag = LAG)

    eval_res = evaluate(y_true, y_pred, is_return = True)
    print(eval_res)

if __name__ == "__main__":
    pipeline("AAPL")

import pandas as pd
import matplotlib.pyplot as plt
from model_utils import load_data, get_train_test, evaluate
import tensorflow_probability as tfp
tfpl = tfp.layers
tfd = tfp.distributions
import tensorflow as tf
tfkl = tf.keras.layers
from sklearn.preprocessing import StandardScaler

def get_model(num_input_feat, lr = 1e-4):
    reg = tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-4)
    model = tf.keras.Sequential([
        tfkl.Dense(
            2**8,
            input_shape=(num_input_feat,),
            kernel_regularizer=reg,
            kernel_initializer='random_normal',
            bias_initializer='zeros'),
        tfkl.Dense(2**4, kernel_regularizer=reg,
                   kernel_initializer='random_normal',
                   bias_initializer='zeros' ),
        tfkl.Dense(2,
                   kernel_initializer='random_normal',
                   bias_initializer='zeros'),
        tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1], scale=1e-6 + tf.math.abs(t[..., 1:])))
    ])
    negloglik = lambda y_true, y_pred: -y_pred.log_prob(y_true)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=negloglik)
    return model

def train_model(X, y, val_size = 0.05):
    num_input_feat = X.shape[1]
    model = get_model(num_input_feat)
    # get callback
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=20, min_delta=1e-1, mode='auto',
                                                baseline=None, restore_best_weights=True)
    # get train and val
    N_val = int(len(X) * val_size)
    X_train, y_train = X[:-N_val, :], y[:-N_val]
    X_val, y_val = X[-N_val:, :], y[-N_val:]
    history = model.fit(X_train,
              y_train,
              validation_data=(X_val, y_val),
              epochs=256,
              verbose=False, shuffle=True,
              callbacks=[callback],
              batch_size=256)

    plot_history(history)
    return model

def plot_history(history):
    df = pd.DataFrame({
        "loss" : history.history["loss"],
        "val_loss": history.history["val_loss"],
    })
    df.plot()
    plt.show()

def get_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def pipeline(company_symbol):
    df = load_data(company_symbol)
    df_train, df_test = get_train_test(df)
    del df

    # train model
    X_train = df_train.drop("target", axis=1).values
    y_train = df_train["target"].values

    # scale data
    scaler = get_scaler(X_train)
    X_train_scld = scaler.transform(X_train)

    # train model
    model = train_model(X_train_scld, y_train, val_size=0.05)

    # evaluate model
    X_test = df_test.drop("target", axis=1).values
    y_test = df_test["target"].values
    X_test_scld = scaler.transform(X_test)

    y_pred = model(X_test_scld).mean().numpy().reshape((-1, ))

    eval_res = evaluate(y_test, y_pred, is_return = True)
    print(eval_res)

if __name__ == "__main__":
    pipeline("AAPL")

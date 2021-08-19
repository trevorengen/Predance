import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def build_train_test_data(coin, test_len=0.2):
    data = 'coin_prices/coin_' + coin + '.csv'
    df = pd.read_csv(data)
    split_row = len(df) - int(test_len * len(df))
    train_data = df.iloc[split_row:]
    test_data = df.iloc[:split_row]

    traincol = train_data.iloc[:, 7:8].values
    testcol = test_data.iloc[:, 7:8].values
    train_shaped = np.reshape(traincol, (-1,1))
    test_shaped = np.reshape(testcol, (-1,1))

    return train_shaped, test_shaped

def scale_data(fit_to, transform_this):
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(fit_to)
    return sc.transform(transform_this)

def reverse_scale(fit_from, transform_this):
    sc = MinMaxScaler(feature_range=(0, 1))
    sc.fit(fit_from)
    return sc.inverse_transform(transform_this)

def make_timesteps(data, n=50):
    X = []
    y = []

    for i in range(n, len(data)):
        X.append(data[i-n:i,0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping

def init_model(inp):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(inp.shape[1], inp.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(optimizer='adam', loss='mae')
    return model
    
def train_and_save(model, X, y, coin, e=500, batchs=12, val_split=0.1):
    callback = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.fit(X, y, epochs=e, batch_size=batchs, validation_split=val_split, callbacks=[callback], verbose=1)
    model.save('saved_models/' + coin)


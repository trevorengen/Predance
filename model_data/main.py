from mmodel import init_model, train_and_save
from hdata import build_train_test_data, make_timesteps, reverse_scale, scale_data
from sklearn.model_selection import train_test_split
from keras.models import load_model
from matplotlib import pyplot
import numpy as np

if __name__=='__main__':
    coin = input('Enter coin (case sensitive): ')
    train_data, test_data = build_train_test_data(coin)
    train_scaled = scale_data(train_data, train_data)
    X, y = make_timesteps(train_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = init_model(X_train)
    train_or_load = input('1 - To train model on data\n2 - To attempt to load.\n')
    if train_or_load == str(1):
        train_and_save(model, X_train, y_train, coin, e=500, batchs=10)
    else:
        model = load_model('saved_models/' + coin)

    test_scaled = scale_data(train_data, test_data)
    new_X, new_y = make_timesteps(test_scaled)
    new_y = np.reshape(new_y, (-1,1))

    model_prediction = model.predict(new_X)
    fix_X_scale = reverse_scale(train_data, model_prediction)
    fix_y_scale = reverse_scale(train_data, new_y)

    i = 1500
    j = i + 300

    fig, ax = pyplot.subplots()
    ax.plot(fix_X_scale[i:j], label='Predictions')
    ax.plot(fix_y_scale[i:j], label='Actual')
    ax.set_title(f'{coin} Price Prediction vs. Actual')
    ax.set_ylabel('Price [USD]')
    ax.set_xlabel('Days')
    ax.legend()
    pyplot.show()

    print(fix_y_scale[i] - fix_X_scale[i])
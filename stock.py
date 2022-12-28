import math
import datetime as dt
import time
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


def get_predict(stock):
    end_date = dt.datetime.now()
    start_date = dt.datetime(2015, 1, 1)

    df = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)

    data = df.filter(['Close'])
    dataset = data.values
    training_len = math.ceil(len(dataset) * 0.8)
        # print(training_len)

        # sclaing
    sclr = MinMaxScaler(feature_range=(0, 1))
    scaled_data = sclr.fit_transform(dataset)

    train = scaled_data[0:training_len, :]
    x_train = []
    y_train = []

    for i in range(60, len(train)):
        x_train.append(train[i - 60:i, 0])
        y_train.append(train[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Training the ML model ...")
    model.fit(x_train, y_train, batch_size=10, epochs=10)

    test = scaled_data[training_len - 60:, :]
    x_test = []
    y_test = dataset[training_len:, :]
    for i in range(60, len(test)):
        x_test.append(test[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = sclr.inverse_transform(predictions)

        # rmse = np.sqrt(np.mean(predictions - y_test)**2)
        # print('rmse: ', rmse)

    train_data = data[:training_len]
    valid = data[training_len:]
    valid['Predictions'] = predictions

        # print(valid)

    last_60 = data[-60:].values
    last_60_scaled = sclr.transform(last_60)
    new_x_test = []
    new_x_test.append(last_60_scaled)
    new_x_test = np.array(new_x_test)
    new_x_test = np.reshape(new_x_test, (new_x_test.shape[0], new_x_test.shape[1], 1))
    pred = model.predict(new_x_test)
    pred = sclr.inverse_transform(pred)
    print(f"Predicted price for {stock} is ${pred[0][0]}")

    plt.figure(figsize=(16, 8))
    plt.title(stock)
    plt.xlabel('Date')
    plt.ylabel('Close price')
    plt.plot(train_data['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'])
    plt.show()

    # except:
    #     print("Please enter a valid stock.")
    #     s = input()
    #     get_predict(s)



get_predict(input())
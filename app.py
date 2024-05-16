import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

np.random.seed(4)

# DATASET
dataset = pd.read_csv('apple_stocks.csv', index_col='Date', parse_dates=['Date'])

# DATA PRE-PROCESSING
# Training and validation sets.
# The LSTM will be trained with data prior to 2016. Validation will be done with data after 2017.
# In both cases, only the highest value of the action for each day will be used.
training_set = dataset[:'2016'].iloc[:,1:2]
validation_set = dataset['2017':].iloc[:,1:2]

training_set['High'].plot(legend=True)
validation_set['High'].plot(legend=True)
plt.ylabel('Stock value ($ USD)')
plt.xlabel('Time (years)')
plt.title('Model Schema')
plt.legend(['Training (2006-2016)', 'Validation (2017 - )'])
plt.show()

# Data normalization
scaled = MinMaxScaler(feature_range=(0,1))
scaled_training_set = scaled.fit_transform(training_set)

# Adjusting training and validation sets.
# The LSTM network will have consecutive data as input "time_step", and 1 data as output (the prediction from that "time_step" data). 
# The training set will be formed in this way.
time_step = 7
X_train = []
Y_train = []
m = len(scaled_training_set)

for i in range(time_step,m):
    # X: "time_step" blocks: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(scaled_training_set[i-time_step:i,0])

    # Y: Next data
    Y_train.append(scaled_training_set[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# CREATION AND TRAINING OF THE LSTM NETWORK
input_data_size = (X_train.shape[1],1)
output_data_size = 1
neurons = 100

model = Sequential()
model.add(LSTM(units=neurons, input_shape=input_data_size))
model.add(Dropout(0.2))  # Regularization with a 20% Dropout ratio to avoid overfitting
model.add(Dense(units=output_data_size))
model.compile(optimizer='rmsprop', loss='mse')
model.fit(X_train, Y_train, epochs=20, batch_size=32)

# STOCK VALUE PREDICTION
x_test = validation_set.values
x_test = scaled.transform(x_test)

X_test = []
validation_dates = []
for i in range(time_step,len(x_test)):
    X_test.append(x_test[i-time_step:i,0])
    validation_dates.append(validation_set.index[i])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

prediction = model.predict(X_test)
prediction = scaled.inverse_transform(prediction)

def prediction_graph(real, prediction, validation_dates):
    plt.plot(validation_dates[0:len(prediction)], real[0:len(prediction)], color='red', label='Real stock value')
    plt.plot(validation_dates[0:len(prediction)], prediction, color='black', label='Stock value prediction')
    plt.ylim(1.1 * np.min(prediction) / 2, 1.1 * np.max(prediction))
    plt.xlabel('Time (months)')
    plt.ylabel('Stock value ($ USD)')
    plt.title('Apple Stock Prediction')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

prediction_graph(validation_set.values, prediction, validation_dates)

# Author -> MVS - Manuel Vallejo Sabadell (MVS-99 github)
# Importing libraries to be used in the microassignment.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import Sequential
from tensorflow import keras



# Import the CSV data set into a pandas dataframe.
df_train = pd.read_csv('Oilprices.csv')

# Remove the rows without a price value
# (i.e., rows where the 'DCOILWTICO' column has a '.' value)
df_train = df_train[df_train['DCOILWTICO'] !='.']

# Extract only the 'DCOILWTICO' column
training_set = df_train.iloc[:, 1:2].values

# Scale the data for better convergence
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Use the first 1000 samples for training and the rest for validation
# Sliding window approach with 60 samples each
#=====DEF > SLIDING WINDOW=====#
# A window of 60 oil prices is selected as input and the next oil price is set as output
# The window is then slid one step forward and the process is repeated to create a series of overlapping sequences
# These sequences are used to train the model to predict future oil prices based on past trends
X_train = []
y_train = []
for i in range(60, 1000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the RNN
def build_rnn_gpu(X_train):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dense(units=1))
    return model

model = build_rnn_gpu(X_train)

# Compile the RNN using the 'adam' optimizer -> stochastic gradient descent optimization algorithm 
# And mean squared error as the loss function
# Loss function measures how well the model is able to predict the expected output
# and is used to update the model's weights during training
# MSE = 1/n * sum((y_pred - y_actual)^2)
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the RNN  -- Batch Size and Epochs are the common recommended ones
#==========DEF > Batch Size==========#
# Batch size determines the number of data samples that are processed at once during training in the neural network.
# During each training iteration, a batch of data samples is passed forward through the network to generate predictions,
# and then backward through the network to compute the gradients and update the model's weights.
# It affects the speed, stability, and generalization of the model's training.
#==========DEF > Epochs==============#
# The number of epochs is a hyperparameter that controls how many times the model will iterate over the dataset.
# Increasing epochs can improve accuracy but may lead to overfitting, while too few epochs can result in an underfit model.
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Plot the training and validation loss over the epochs
loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('traininandvalidation_loss_over_epochs.png')

# Finally, show the performance with the test data set.
# 1º Load the test data
data_test = df_train.iloc[1001:, 1:2].values
data_test = sc.transform(data_test)

X_test = []
for j in range(60, len(data_test)):
    X_test.append(training_set_scaled[j-60:j])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# 2º Use the trained model to make predictions on the test data
# In reshape function:
# -1 -> Placeholder: Whatever number of samples there are, keep as it is
# 60 -> Time steps in the input sequence (sequence of data as indicated in training)
# 1 -> Number of features of each step (in this case there is only 1 anyway)
predictions = model.predict(X_test, batch_size = 32)

# 3º Inverse transform the predicted data to obtain the original price values
predictions = sc.inverse_transform(predictions)

# Create dates_test variable
dates_test = df_train.iloc[1001+60:1001+len(predictions), 0].values

tick_locations = np.arange(0, len(dates_test), 30)
tick_labels = [dates_test[i] for i in tick_locations]


max_y_ticks = max(np.max(data_test[0]), np.max(data_test[1]))
min_y_ticks = min(np.min(data_test[0]), np.min(data_test[1]))
coeff_y_ticks = max_y_ticks/min_y_ticks


# Plot the predicted oil prices for the test data
plt.figure(figsize=(12,8))
plt.plot(dates_test, predictions, label='Predicted')
plt.scatter(df_train['DATE'][1001:1254], df_train['DCOILWTICO'][1001:1254], color = "r", label='Actual')
plt.title('Predicted vs Actual Oil Prices (Test Data)')
plt.xlabel('Date')
plt.ylabel('Oil Price (USD)')
plt.yticks(np.arange(min_y_ticks, max_y_ticks, coeff_y_ticks))
plt.xticks(np.arange(0, len(dates_test[0]) + len(dates_test[1]), 5), rotation = 45)
plt.legend()
plt.savefig('final_plot_predicted_prices')
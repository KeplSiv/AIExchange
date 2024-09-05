# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM
# import matplotlib.pyplot as plt


# def load_data(file_path):
#     df = pd.read_csv(file_path, header=None, parse_dates=[0], index_col=0)
#     return df


# # Load exchange rate and inflation rate data
# exchange_rate_data = load_data('exchange_rate.csv')
# inflation_rate_data = load_data('inflation_data.csv')

# # Ensure both datasets have the same time index
# combined_data = exchange_rate_data.join(
#     inflation_rate_data, how='inner', lsuffix='_exchange', rsuffix='_inflation')
# combined_data.dropna(inplace=True)  # Drop rows with missing values, if any

# # Normalize the data
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(combined_data)


# def create_sequences(data, sequence_length):
#     sequences = []
#     for i in range(len(data) - sequence_length):
#         sequence = data[i:i + sequence_length]
#         sequences.append(sequence)
#     return np.array(sequences)


# sequence_length = 11  # for monthly data, a year sequence
# sequences = create_sequences(scaled_data, sequence_length)

# X = sequences[:, :-1, :]
# y = sequences[:, -1, 0]  # predicting the next month's exchange rate

# # Split into training and testing sets
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Define the model
# model = Sequential([
#     LSTM(50, activation='relu', input_shape=(sequence_length - 1, 2)),
#     Dense(1)
# ])

# model.compile(optimizer='adam', loss='mse')

# # Train the model
# history = model.fit(X_train, y_train, epochs=300,
#                     validation_data=(X_test, y_test))

# # Evaluate the model
# loss = model.evaluate(X_test, y_test)
# print(f'Test Loss: {loss}')

# # Make predictions
# predictions = model.predict(X_test)

# # Inverse transform the predictions to get them back to the original scale
# predictions_inverse = scaler.inverse_transform(np.concatenate(
#     (predictions, X_test[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]

# # Plot actual vs predicted values
# plt.plot(combined_data.index[train_size + sequence_length:],
#          combined_data.iloc[train_size + sequence_length:, 0], label='Actual')
# plt.plot(combined_data.index[train_size + sequence_length:],
#          predictions_inverse, label='Predicted')
# plt.legend()
# plt.show()

# # Example: Predicting the next month's exchange rate based on the last 12 months of data
# input_sequence = scaled_data[-sequence_length:
#                              ].reshape((1, sequence_length, 2))

# # Make the prediction
# predicted_exchange_rate = model.predict(input_sequence)

# # Inverse transform the prediction to get the original scale
# predicted_exchange_rate_original = scaler.inverse_transform(np.concatenate(
#     (predicted_exchange_rate, input_sequence[:, -1, 1].reshape(-1, 1)), axis=1))[:, 0]
# print(
#     f'Predicted Exchange Rate for Next Month: {predicted_exchange_rate_original[0]}')

# model.save('lstm_exchange_rate_model.h5')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


def load_data(file_path):
    df = pd.read_csv(file_path, header=None, parse_dates=[0], index_col=0)
    return df


# Load exchange rate and inflation rate data
exchange_rate_data = load_data('exchange_rate.csv')
inflation_rate_data = load_data('inflation_data.csv')

# Ensure both datasets have the same time index
combined_data = exchange_rate_data.join(
    inflation_rate_data, how='inner', lsuffix='_exchange', rsuffix='_inflation')
combined_data.dropna(inplace=True)  # Drop rows with missing values, if any

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data)


def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)


sequence_length = 11  # for monthly data, a year sequence
sequences = create_sequences(scaled_data, sequence_length)

X = sequences[:, :-1, :]
# predicting the next month's exchange rate and inflation rate
y = sequences[:, -1, :]

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length - 1, 2)),
    Dense(2)  # Output layer with 2 neurons, one for each prediction (exchange rate and inflation rate)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=300,
                    validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions to get them back to the original scale
predictions_inverse = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test)

# Plot actual vs predicted values for exchange rate
plt.figure(figsize=(14, 5))
plt.plot(combined_data.index[train_size + sequence_length:],
         y_test_inverse[:, 0], label='Actual Exchange Rate')
plt.plot(combined_data.index[train_size + sequence_length:],
         predictions_inverse[:, 0], label='Predicted Exchange Rate')
plt.legend()
plt.title('Exchange Rate: Actual vs Predicted')
plt.show()

# Plot actual vs predicted values for inflation rate
plt.figure(figsize=(14, 5))
plt.plot(combined_data.index[train_size + sequence_length:],
         y_test_inverse[:, 1], label='Actual Inflation Rate')
plt.plot(combined_data.index[train_size + sequence_length:],
         predictions_inverse[:, 1], label='Predicted Inflation Rate')
plt.legend()
plt.title('Inflation Rate: Actual vs Predicted')
plt.show()

# Example: Predicting the next month's exchange rate and inflation rate based on the last 12 months of data
input_sequence = scaled_data[-sequence_length:
                             ].reshape((1, sequence_length, 2))

# Make the prediction
predicted_values = model.predict(input_sequence)

# Inverse transform the prediction to get the original scale
predicted_values_original = scaler.inverse_transform(predicted_values)

predicted_exchange_rate_original = predicted_values_original[0, 0]
predicted_inflation_rate_original = predicted_values_original[0, 1]

print(
    f'Predicted Exchange Rate for Next Month: {predicted_exchange_rate_original}')
print(
    f'Predicted Inflation Rate for Next Month: {predicted_inflation_rate_original}')

model.save('lstm_exchange_rate_inflation_model.h5')

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
# import datetime

# # Load the saved model
# model = tf.keras.models.load_model('lstm_exchange_rate_model.h5')

# # Load data function without column names


# def load_data(file_path):
#     df = pd.read_csv(file_path, parse_dates=[0], index_col=0, header=None)
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

# # Function to create sequences


# def create_sequences(data, sequence_length):
#     sequences = []
#     for i in range(len(data) - sequence_length):
#         sequence = data[i:i + sequence_length]
#         sequences.append(sequence)
#     return np.array(sequences)


# # for monthly data, a year sequence (1 less than total to match model input)
# sequence_length = 11

# # Function to predict exchange rate and inflation rate for a given year and month


# def predict_for_year_month(year, month):
#     # Calculate the number of months to predict
#     last_date = combined_data.index[-1]
#     future_date = pd.Timestamp(datetime.date(year, month, 1))
#     months_to_predict = (future_date.year - last_date.year) * \
#         12 + (future_date.month - last_date.month)

#     if months_to_predict <= 0:
#         raise ValueError(
#             "The specified year and month should be in the future")

#     # Start with the last available data
#     input_sequence = scaled_data[-sequence_length:
#                                  ].reshape((1, sequence_length, 2))

#     # Predict month by month
#     predictions = []
#     for _ in range(months_to_predict):
#         # Use only the last 10 elements for the input to the model
#         model_input_sequence = input_sequence[:, -10:, :]
#         predicted_values = model.predict(model_input_sequence)
#         predictions.append(predicted_values[0, 0])
#         # Update the input sequence with the new prediction
#         next_input = np.array(
#             [[predicted_values[0, 0], input_sequence[0, -1, 1]]])
#         next_input_scaled = scaler.transform(next_input)
#         input_sequence = np.concatenate(
#             (input_sequence[:, 1:, :], next_input_scaled.reshape(1, 1, 2)), axis=1)

#     # Prepare the array for inverse transform
#     predicted_array = np.array(predictions).reshape(-1, 1)
#     last_inflation_value = np.full(
#         (predicted_array.shape[0], 1), input_sequence[0, -1, 1])
#     combined_predictions = np.concatenate(
#         (predicted_array, last_inflation_value), axis=1)

#     # Inverse transform the prediction to get the original scale
#     predicted_exchange_rate = scaler.inverse_transform(combined_predictions)[
#         :, 0]

#     return predicted_exchange_rate[-1]


# # Example usage
# year = 2023
# month = 1
# predicted_exchange_rate = predict_for_year_month(year, month)
# print(
#     f'Predicted Exchange Rate for {year}-{month:02d}: {predicted_exchange_rate}')


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime

# Load the saved model
model = tf.keras.models.load_model('lstm.h5')

# Load data function without column names


def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=[0], index_col=0, header=None)
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

# Function to create sequences


def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)


# for monthly data, a year sequence (1 less than total to match model input)
sequence_length = 11

# Function to predict exchange rate and inflation rate for a given year and month


def predict_for_year_month(year, month):
    # Calculate the number of months to predict
    last_date = combined_data.index[-1]
    future_date = pd.Timestamp(datetime.date(year, month, 1))
    months_to_predict = (future_date.year - last_date.year) * \
        12 + (future_date.month - last_date.month)

    if months_to_predict <= 0:
        raise ValueError(
            "The specified year and month should be in the future")

    # Start with the last available data
    input_sequence = scaled_data[-sequence_length:
                                 ].reshape((1, sequence_length, 2))

    # Predict month by month
    exchange_rate_predictions = []
    inflation_rate_predictions = []
    for _ in range(months_to_predict):
        # Use only the last 10 elements for the input to the model
        model_input_sequence = input_sequence[:, -10:, :]
        predicted_values = model.predict(model_input_sequence)
        exchange_rate_predictions.append(predicted_values[0, 0])
        inflation_rate_predictions.append(input_sequence[0, -1, 1])
        # Update the input sequence with the new prediction
        next_input = np.array(
            [[predicted_values[0, 0], input_sequence[0, -1, 1]]])
        next_input_scaled = scaler.transform(next_input)
        input_sequence = np.concatenate(
            (input_sequence[:, 1:, :], next_input_scaled.reshape(1, 1, 2)), axis=1)

    # Prepare the arrays for inverse transform
    exchange_rate_predictions_array = np.array(
        exchange_rate_predictions).reshape(-1, 1)
    inflation_rate_predictions_array = np.array(
        inflation_rate_predictions).reshape(-1, 1)
    combined_predictions = np.concatenate(
        (exchange_rate_predictions_array, inflation_rate_predictions_array), axis=1)

    # Inverse transform the predictions to get the original scale
    predicted_values = scaler.inverse_transform(combined_predictions)
    predicted_exchange_rate = predicted_values[:, 0]
    predicted_inflation_rate = predicted_values[:, 1]

    return predicted_exchange_rate[-1], predicted_inflation_rate[-1]


# Example usage
year = 2023
month = 1
predicted_exchange_rate, predicted_inflation_rate = predict_for_year_month(
    year, month)
print(
    f'Predicted Exchange Rate for {year}-{month:02d}: {predicted_exchange_rate}')
print(
    f'Predicted Inflation Rate for {year}-{month:02d}: {predicted_inflation_rate}')

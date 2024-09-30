# Exchange Rate and Inflation Rate Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict the exchange rate and inflation rate for the next month based on the previous year's data. 
The model is trained using historical data, and predictions are made for both exchange rates and inflation rates.

## Project Structure

- `exchange_rate.csv`: Contains the historical exchange rate data.
- `inflation_data.csv`: Contains the historical inflation rate data.
- `lstm_exchange_rate_inflation_model.h5`: The trained LSTM model saved after training.

## Prerequisites

Before you start, ensure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib bash
```


## How to Run the Code
1.`Load the Data`
The project loads two CSV files: one containing exchange rate data and the other containing inflation rate data. Both files should have a date as the first column and corresponding values in the second column.

```python

def load_data(file_path):
    df = pd.read_csv(file_path, header=None, parse_dates=[0], index_col=0)
    return df
```

2.`Data Preprocessing`
The two datasets are joined on their date index. Any missing values are removed, and the data is normalized using MinMaxScaler to transform the values into the range [0, 1].

```python
# Ensure both datasets have the same time index
combined_data = exchange_rate_data.join(
    inflation_rate_data, how='inner', lsuffix='_exchange', rsuffix='_inflation')
combined_data.dropna(inplace=True)  # Drop rows with missing values, if any
```

- `Normalize the data`
```python
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data)
```

3.`Sequence Creation`
The create_sequences function creates sequences based on the past 12 months (11 months of data, and the current month for prediction). This is necessary for the LSTM model to learn from time-series data.

```python
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

sequence_length = 11  # for monthly data, a year sequence
sequences = create_sequences(scaled_data, sequence_length)
``` 
4.`Model Architecture`
The model is defined using Keras's Sequential API. It consists of an LSTM layer followed by a Dense output layer with 2 neurons—one for predicting the exchange rate and the other for predicting the inflation rate.

```python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length - 1, 2)),
    Dense(2)  # Output layer with 2 neurons, one for each prediction
])
```

5.`Training the Model`
The model is compiled using the adam optimizer and mean squared error (mse) as the loss function. The training is done over 300 epochs, with 20% of the data used for validation.

```python
model.compile(optimizer='adam', loss='mse')

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the model
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test))
``` 
6.`Making Predictions`
After training, the model is evaluated on the test data. Predictions are made, and the results are compared with the actual values for both the exchange rate and inflation rate.

```python
# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions to original scale
predictions_inverse = scaler.inverse_transform(predictions)
y_test_inverse = scaler.inverse_transform(y_test)
```

7.`Example Prediction for Next Month`
You can use the model to predict the next month’s exchange rate and inflation rate based on the last 12 months of available data.

```python

# Predicting the next month's exchange rate and inflation rate
input_sequence = scaled_data[-sequence_length:].reshape((1, sequence_length, 2))
predicted_values = model.predict(input_sequence)

# Inverse transform the prediction to get the original scale
predicted_values_original = scaler.inverse_transform(predicted_values)

predicted_exchange_rate_original = predicted_values_original[0, 0]
predicted_inflation_rate_original = predicted_values_original[0, 1]

print(f'Predicted Exchange Rate for Next Month: {predicted_exchange_rate_original}')
print(f'Predicted Inflation Rate for Next Month: {predicted_inflation_rate_original}')
```

8.`Saving the Model`
The trained model is saved in .h5 format for future use.

```python
model.save('lstm_exchange_rate_inflation_model.h5')
```

9.`Visualizing Results`
The results of the model’s predictions are visualized using matplotlib for both exchange rate and inflation rate. These plots display actual values versus predicted values for the test dataset.

```python
import matplotlib.pyplot as plt

# Plot actual vs predicted values for exchange rate
plt.figure(figsize=(14, 5))
plt.plot(combined_data.index[train_size + sequence_length:], y_test_inverse[:, 0], label='Actual Exchange Rate')
plt.plot(combined_data.index[train_size + sequence_length:], predictions_inverse[:, 0], label='Predicted Exchange Rate')
plt.legend()
plt.title('Exchange Rate: Actual vs Predicted')
plt.show()

# Plot actual vs predicted values for inflation rate
plt.figure(figsize=(14, 5))
plt.plot(combined_data.index[train_size + sequence_length:], y_test_inverse[:, 1], label='Actual Inflation Rate')
plt.plot(combined_data.index[train_size + sequence_length:], predictions_inverse[:, 1], label='Predicted Inflation Rate')
plt.legend()
plt.title('Inflation Rate: Actual vs Predicted')
plt.show()
```

# Example Plot
<img width="885" alt="Screenshot 2024-09-30 at 11 17 18 AM" src="https://github.com/user-attachments/assets/90633144-d5e6-4122-89c6-d171a1f19f7c">

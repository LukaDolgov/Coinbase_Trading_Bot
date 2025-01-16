import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta, timezone
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import model_from_json
import json
import joblib

# 1. Load and preprocess the data
# Replace 'your_data.csv' with your stock price data file

#how many days to shift to avoid lag
SHIFT_PREDICTION = 4

class RNN_model():
    def __init__(self):
        pass

def shift_left(array):
    if array.size > 0:  # Check if array is non-empty
        return np.roll(array, -1)  # Roll array left by 1 position





class LSTM_model():
    def __init__(self, prevmodel, traineddata, oldscaler):
        self.model = prevmodel
        self.traindata = traineddata
        self.scaler = oldscaler
    def generate_model(self):
        # 1. Load the data
        # Assuming your data is a 2D NumPy array where each row is [time, low, high, open, close, volume]
        data = np.array([
            # Example candles: [time, low, high, open, close, volume]
            self.traindata
        ])
        # Remove the batch dimension, shape becomes (900, 6)
        data_2d = data[0]
        data_2d = np.flip(data_2d, axis=0)
        print(data_2d.shape)
        print(data_2d[0][4])
        # Convert to DataFrame for easier handling
        columns = ['time', 'low', 'high', 'open', 'close', 'volume']
        df = pd.DataFrame(data_2d, columns=columns)
        # Drop the 'time' column (not useful for prediction)
        df = df.drop(columns=['time'])
        # 2. Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        joblib.dump(scaler, "scaler.pkl") #save scalar conversion
        print(df.shape)
        # 3. Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])  # Use all features
                y.append(data[i + seq_length, 3])  # Predict the 'close' price
            return np.array(X), np.array(y)

        SEQ_LENGTH = 7  # Sequence length, amount of days before prediction
        X, y = create_sequences(scaled_data, SEQ_LENGTH)

        # Split into training and testing sets
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 4. Build the LSTM model
        model = Sequential([
            LSTM(20, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
            Dropout(0.2),
            LSTM(20),
            Dropout(0.2),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # 5. Train the model
       # early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
           # callbacks=[early_stop]
        )
        # 6. Make predictions
        predicted_prices = model.predict(X_test)
        print(predicted_prices)
        print(predicted_prices.shape)
        # Transform predictions and actual values back to original scale
        predicted_prices = scaler.inverse_transform(
            np.hstack([np.zeros((len(predicted_prices), df.shape[1] - 1)), predicted_prices])
        )[:, -1]  # Extract the predicted 'close' price


        actual_prices = scaler.inverse_transform(
            np.hstack([np.zeros((len(y_test), df.shape[1] - 1)), y_test.reshape(-1, 1)])
        )[:, -1]
        #shifting whole predictions to 3 days ahead (more accuracy)        
        for i in range(SHIFT_PREDICTION):
            predicted_prices = shift_left(predicted_prices)
        predicted_prices = predicted_prices[:-SHIFT_PREDICTION]
        actual_prices = actual_prices[:-SHIFT_PREDICTION]
        print("predicted prices")
        print(predicted_prices)
        self.model = model
        print("actual prices: ")
        print(actual_prices)
        #statistics:
        mse_test = mean_squared_error(actual_prices, predicted_prices)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(actual_prices, predicted_prices)
        r2_test = r2_score(actual_prices, predicted_prices)
        print("MSE (TEST): ")  
        print(mse_test)
        print("MAE (TEST): ") 
        print(mae_test)
        print("r2 (TEST): ") 
        print(r2_test)
        
        #save model:
        
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # Save the weights to HDF5
        model.save_weights("model_weights.weights.h5")
        with open("training_data.json", "w") as json_file:
            json.dump(self.traindata, json_file, indent=4)
        
        # 7. Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(actual_prices, label='Actual Prices')
        plt.plot(predicted_prices, label='Predicted Prices')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
    def use_model(self, recent_data):
        """
        Use the trained model to predict the next price based on the last 5 days of data.
        
        Parameters:
            recent_data (np.array): A NumPy array of shape (7, num_features) containing the last 5 days of data.
            7 rows

        Returns:
            float: Predicted price for the price in 4 days.
        """
        # Ensure the recent_data is scaled similarly to the training data
        scaler = self.scaler
        # Use the training data to fit the scaler
        scaled_recent_data = scaler.transform(recent_data)
        scaled_recent_data = scaled_recent_data.reshape(1, recent_data.shape[0], recent_data.shape[1])

        # Make the prediction
        predicted_price_scaled = self.model.predict(scaled_recent_data)
        # Transform the predicted price back to the original scale
        predicted_price = scaler.inverse_transform(
            np.hstack([np.zeros((1, recent_data.shape[1] - 1)), predicted_price_scaled])
        )[:, -1]
        print("price in three days " + f"{predicted_price}")
        return predicted_price[0]

        


#for testing:

def fetch_candles(product_id, start_time, end_time, granularity=60):
    """
    Fetch historical candle data for a given product and time range.
    """
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "granularity": granularity
    }
    headers = {"Accept": "application/json"}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        return response.json()  # List of candles
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []

def historic_simul(timeframe, days):
   # Fetch historical candles in chunks of 300 days.
    totallist = []
    end_time = datetime.now()
    # Loop to fetch data in 300-day chunks
    if timeframe == "1D":
        for i in range(0, (days // 300) + (1 if days % 300 != 0 else 0)):
            start_time = end_time - timedelta(days=300)
            # Fetch candles and append to the list
            candles = fetch_candles("BTC-USD", start_time, end_time, 86400)
            totallist.append(candles)
            # Update end_time for the next iteration
            end_time = start_time
        # Flatten the list of lists
        flattened_list = [item for sublist in totallist for item in sublist]
        print(len(flattened_list))
    return flattened_list





prevmod = 0
prevweights = 0
#update in days sets of 300

#testing
#LSTM_DAY_TRACKER = LSTM_model(prevmod, historic_simul("1D", 900), 0)
#print(LSTM_DAY_TRACKER.traindata)
#print(LSTM_DAY_TRACKER.generate_model())
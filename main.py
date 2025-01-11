#edit
from coinbase_advanced_trader.enhanced_rest_client import EnhancedRESTClient
from datetime import datetime, timedelta, timezone
import time
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from classes import user, candle, check_orders, UserLimitOrder, UserBuyLimitOrder, UserSellLimitOrder, UserStopSellLimitOrder
from strategies import strategy1
from neuralnets import LSTM_model

import json
import requests


API_KEY = "organizations/a53e9113-6e00-4977-9ed9-39625d25e778/apiKeys/9b08db99-7161-4c48-8284-a40896be0f7e"
PRIV_KEY = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIK22c8FX5bQDFObeJN4HezU+qZdaq1GfAB5t3oZdz8huoAoGCCqGSM49\nAwEHoUQDQgAEIN4XUIofwHAC1bweFaNr3M4tPgpsC2kNr/OT91l/J+1nT5v+aNt/\njgbbnjWMuyd25m9OHmfciD/y6YMEHjnLTw==\n-----END EC PRIVATE KEY-----\n"
client = EnhancedRESTClient(api_key=API_KEY, api_secret=PRIV_KEY)

#change product here
PRODUCT = "BTC-USD"
STRATUSE = strategy1()
LENGTH_PER_MIN = 5


def set_time_to_utc_minus_5():
    # Get the current UTC time
    now_utc = datetime.now(timezone.utc)
    # Adjust to UTC-5
    utc_minus_5 = now_utc - timedelta(hours=5)
    # Format the adjusted time
    formatted_time = utc_minus_5.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Current time adjusted to UTC-5: {formatted_time}")
def return_milliseconds():
    now_utc = datetime.now(timezone.utc)
    utc_minus_5 = now_utc - timedelta(hours=5)
    milliseconds = int(utc_minus_5.strftime('%f')[:3]) 
    return milliseconds

def generate_low(values):
    return min(values)
def generate_high(values):
    return max(values)
def generate_figure(min_counter, lows, highs, opens, closes):
    data = {
    "Counter": min_counter,
    "Open": opens,
    "High": highs,
    "Low": lows,
    "Close": closes,
    }  
    # Convert data into a DataFrame
    df = pd.DataFrame(data)

    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['Counter'],  # X-axis (e.g., date or time)
        open=df['Open'],  # Opening price
        high=df['High'],  # Highest price
        low=df['Low'],  # Lowest price
        close=df['Close'],  # Closing price
        increasing_line_color='green',  # Color for bullish candles
        decreasing_line_color='red'  # Color for bearish candles
    )])

    # Customize the layout
    fig.update_layout(
        title="Candlestick Chart BTC from execution",
        xaxis_title="Minutes since execution",
        yaxis_title="Price",
        template="plotly_dark",  # Optional dark theme
        width=1000,
        height=600
    )

    # Show the chart
    fig.show()
def dynamic_candlestick_chart(min_counter, lows, highs, opens, closes):
    #close previous
    plt.close('all')
    """
    Arguments:
        df (DataFrame): A pandas DataFrame with 'Open', 'High', 'Low', 'Close' columns and a time-based index.
    """
    data = {
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
    }
    start_time = datetime.now()  # Start at the current time
    index = [start_time + timedelta(minutes=minute) for minute in min_counter]
    df = pd.DataFrame(data,  index=pd.DatetimeIndex(index))
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axis
    # Loop to simulate real-time updates
    for i in range(1, len(df) + 1):  
        ax.clear()  # Clear the axis for the new plot
        # Use the subset of data for plotting
        data_to_plot = df.iloc[:i]
        # Plot the candlestick chart
        mpf.plot(data_to_plot, type='candle', ax=ax, style='charles', volume=False)
        # Refresh the plot
        fig.canvas.flush_events()
def fetch_recent_trades(product_id):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/trades"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []
def calculate_minute_volume(trades):
    # Get the start of the current minute in UTC
    current_minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    total_volume = 0.0
    for trade in trades:
        # Parse the trade time
        trade_time = datetime.fromisoformat(trade["time"].replace("Z", "+00:00"))
        # Check if the trade occurred within the current minute
        if trade_time >= current_minute:
            total_volume += float(trade["size"])

    return total_volume

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

def historic_simul(timeframe):
    #granularity is measure of candle, 3600 hour, 600 minute, 60 second, etc.
    # Define start and end times
    end_time = datetime.now()
    start_time = end_time - timedelta(days=14)  # Last 24 hours 
    #3600 for hours should be 72 candles #86400 for day get 14 candles
    if timeframe == "1D":
        historic_candles_1D = fetch_candles(PRODUCT, start_time, end_time, 86400)
        #result is a ordered list of lists of candles in format: [time, low, high, open, close, volume]
    return historic_candles_1D



# retrieve balances and various information
with open("statistics.json", "r") as file:
    data = json.load(file)

order_type_map = {
    "limbuy": UserBuyLimitOrder,
    "limsell": UserSellLimitOrder,
    "stoplimsell": UserStopSellLimitOrder,
}
data["Corderbook"] = [
    order_type_map[order["type"]](0, 0, 0).from_dict(order) if isinstance(order, dict) else order
    for order in data["Corderbook"]
]

# Access the orders
for order in data["Corderbook"]:
    print(order.to_dict())
print("orders: ")
print([order.to_dict() for order in data["Corderbook"]])  # Verify the loaded orders

# Access specific values
#USD balance in client account
USDbalance = data["USDbalance"]
#Crypto balance in client account
CRYPTbalance = data['CRYPTbalance']
#client order book
Corderbook = data["Corderbook"]
#time spent trading
timespent = data["timespent"]

print(f"Balance: {USDbalance}")
print(f"Crypto Balance: {CRYPTbalance}")



#example update data["timespent"] = 100


def main():
    continueRun = True
    #resets every 60 seconds
    seconds_value_array = []
    prevmod=0
    LSTM_DAY_TRACKER = LSTM_model(prevmod, historic_simul("1D"))
    print(LSTM_DAY_TRACKER.inpdata)
    #tracks minutes, never resets as of now
    min_counter = []
    lows = []
    highs = []
    opens = []
    closes = []
    candles = []
    user_trader = user(USDbalance, CRYPTbalance, Corderbook)
    input_test = input("input 0 or 1 for testing mode/normal mode: ")
    print(user_trader.Corderbook)
    #temp vars orders in form percent, current price, quantity (in crypto) 1 to not add, 0 to add
    if int(input_test) == 0:
        product = client.get_product(product_id=PRODUCT)
        user_trader.Corderbook.append(UserBuyLimitOrder(0.01, float(product.price), 0.1))
    print("Corderbook length:", len(user_trader.Corderbook))
    second_counter = 0
    min_increment = 0
    while continueRun:
        #check if milliseconds = 0 to make second time slot
        milliseconds = return_milliseconds()
        if milliseconds == 0:
            set_time_to_utc_minus_5()
            product = client.get_product(product_id=PRODUCT)
            order_book = client.get_product_book(product_id=PRODUCT)
            trades = fetch_recent_trades(product_id=PRODUCT)
            #testing
            print(product.price)
            print("USD " + str(user_trader.USDbalance))
            print("Crypto: " + str(user_trader.CRYPTbalance))
            user_trader = check_orders(product.price, order_book, user_trader)
            #increment + show active orders + testing
            for order in user_trader.Corderbook:
                if order.terminated == False:
                    print(order.to_dict())
            seconds_value_array.append(product.price)
            second_counter += 1
            global timespent
            timespent += 1
            STRATUSE.candles = candles
            user_trader = STRATUSE.execute(product.price, order_book, user_trader)     
            if second_counter >= LENGTH_PER_MIN:
                min_increment += 1
                STRATUSE.currmin = min_increment
                minute_volume = calculate_minute_volume(trades)
                print("minute volume " + str(minute_volume))
                low = float(generate_low(seconds_value_array))
                high = float(generate_high(seconds_value_array))
                #make minute candle
                min_counter.append(min_increment)
                lows.append(low)
                highs.append(high)
                opens.append(float(seconds_value_array[0]))
                closes.append(float(seconds_value_array[len(seconds_value_array) - 1]))
                candles.append(candle(min_increment, low, high, float(seconds_value_array[0]), float(seconds_value_array[len(seconds_value_array) - 1]), minute_volume))
                dynamic_candlestick_chart(min_counter, lows, highs, opens, closes)
                #reset seconds
                seconds_value_array = []
                second_counter = 0
                print("winrate: " + str(STRATUSE.winrate))
                print("updated graph")
                #for testing
                if min_increment >= 5:
                    for order in user_trader.Corderbook:
                        if order.terminated == False:
                            order.cancelled = True
                            print(order.to_dict())
                            print("cancelled order")
                    continueRun = False
                    print("ending winrate: " + str(STRATUSE.winrate))
                    print("ending USDbal: " + str(user_trader.USDbalance))
                    print("ending cryptobal: " + str(user_trader.CRYPTbalance))
                
    close(user_trader)
            
def close(user):
    data["Corderbook"] = [order.to_dict() for order in user.Corderbook] # order.to_dict() for order in user.Corderbook # [] to reset
    data["USDbalance"] = user.USDbalance
    data["CRYPTbalance"] = user.CRYPTbalance
    data["timespent"] = timespent
    with open("statistics.json", "w") as file:
        json.dump(data, file, indent=4)    
                
main()






#getting a balance of all currencies
#balances = client.list_held_crypto_balances()
#print(balances)
#getting a balance of specific currency
#balance = client.get_crypto_balance("USD")
#print(balance)
    
#testing orders
# while balance > 1.0:
   # client.fiat_limit_buy("PEPE-USD", ".7", ".00001")
    
    
    
    
    
# Perform a market buy
#client.fiat_market_buy("BTC-USDC", "10")

#Place a $10 buy order for BTC-USD near the current spot price of BTC-USDC
#client.fiat_limit_buy("BTC-USDC", "10")

#Place a $10 buy order for BTC-USD at a limit price of $10,000
#client.fiat_limit_buy("BTC-USD", "1", "10000")

#Place a $10 buy order for BTC-USD at a 10% discount from the current spot price of BTC-USDC
#client.fiat_limit_buy("BTC-USDC", "10", price_multiplier=".90")

#Place a $10 sell order for BTC-USD at a limit price of $100,000
#client.fiat_limit_sell("BTC-USDC", "10", "100000")

#Place a $10 sell order for BTC-USD near the current spot price of BTC-USDC
#client.fiat_limit_sell("BTC-USDC", "5")

#Place a $10 sell order for BTC-USD at a 10% premium to the current spot price of BTC-USDC
#client.fiat_limit_sell("BTC-USDC", "5", price_multiplier="1.1")
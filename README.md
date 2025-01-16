# Cointbase_Trading_Bot
This is my first attempt at a trading bot using machine learning

I am using an LSTM model to track stock price and my bot will make a decision based off long-trades only. Currently the bot is trained on 900 days of bitcoin data in terms of solely their candles (high low open close volume). It tracks the past week before deciding the future closing price in 3 days (From the current day).

I have replicated coinbase order books with their API because due to my father's work I am unable to trade on a real account and therefore made my own fake brokerage through coinbase which replicates transactions and crypto orderbooks.


 

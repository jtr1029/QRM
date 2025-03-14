import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from arch import arch_model
import matplotlib.pyplot as plt
from config import NEWS_API_KEY  # Import API key

# Ensure nltk resources are available
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function: Fetch News Data from NewsAPI
def fetch_news(query="stock market", language="en"):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json()["articles"]
        return [(a["title"], a["description"], a["url"]) for a in articles]
    return []

# Function: Analyze Sentiment of News Articles
def analyze_sentiment(text):
    if text:
        score = sia.polarity_scores(text)
        return score["compound"]  # Returns sentiment score (-1 to 1)
    return 0

# Function: Fetch Stock Price Data from Yahoo Finance
def fetch_stock_data(ticker="AAPL", period="1mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

# Function: Plot Stock Prices
def plot_stock_price(ticker="AAPL"):
    df = fetch_stock_data(ticker)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label=f"{ticker} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title(f"{ticker} Stock Price Movement")
    plt.legend()
    st.pyplot(plt)

# Function: Calculate GARCH-based Volatility Forecast
def garch_volatility(ticker="AAPL"):
    df = fetch_stock_data(ticker)
    df["Daily Return"] = df["Close"].pct_change().dropna() * 100  # Convert to percentage

    model = arch_model(df["Daily Return"].dropna(), vol="Garch", p=1, q=1)
    res = model.fit(disp="off")

    forecast = res.forecast(start=len(df), horizon=5)  # Forecast next 5 days
    predicted_volatility = forecast.variance.iloc[-1].values
    return predicted_volatility

# Streamlit UI
st.title("📈 News-Based Financial Insights App")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)", "AAPL")

if st.button("Analyze"):
    st.subheader(f"Latest News on {ticker}")

    # Fetch and Display News
    news = fetch_news(ticker)
    for title, desc, url in news[:5]:  # Show top 5 articles
        sentiment = analyze_sentiment(title + " " + desc)
        st.markdown(f"**[{title}]({url})**")
        st.write(f"Sentiment Score: {sentiment:.2f}")
        st.write("---")

    # Stock Price Plot
    st.subheader(f"{ticker} Stock Price Data")
    plot_stock_price(ticker)

    # GARCH Volatility
    volatility = garch_volatility(ticker)
    st.subheader(f"📊 Estimated Volatility for {ticker}:")
    st.write(volatility)

    st.success("✅ Analysis Complete!")

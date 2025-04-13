import yfinance as yf
import pandas as pd
import requests
from alpha_vantage.fundamentaldata import FundamentalData
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newspaper import Article
from bs4 import BeautifulSoup
import json
import feedparser
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import os
from typing import Dict, List
import pickle
import logging

ALPHA_VANTAGE_API_KEY = "9RBLWH3XU1NBWORU"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to collect the most recent stock price data
def get_latest_stock_price(ticker):
    stock = yf.Ticker(ticker)
    latest_price = stock.history(period="1d")
    return latest_price.iloc[-1] if not latest_price.empty else None

# Function to collect historical stock price data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

# Function to collect fundamental data
def get_fundamental_data(ticker):
    fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    balance_sheet, _ = fd.get_balance_sheet_annual(ticker)
    income_statement, _ = fd.get_income_statement_annual(ticker)
    return balance_sheet, income_statement

# Function to collect macroeconomic indicators
def get_macro_data():
    macro_indicators = {}
    
    # Real GDP
    gdp_url = f"https://www.alphavantage.co/query?function=REAL_GDP&apikey={ALPHA_VANTAGE_API_KEY}"
    gdp_response = requests.get(gdp_url)
    macro_indicators['GDP'] = gdp_response.json()
    print("GDP Response:", macro_indicators['GDP'])  # Debug print
    
    # Interest Rates
    rates_url = f"https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&apikey={ALPHA_VANTAGE_API_KEY}"
    rates_response = requests.get(rates_url)
    macro_indicators['Interest Rates'] = rates_response.json()
    print("Interest Rates Response:", macro_indicators['Interest Rates'])  # Debug print
    
    # CPI (Inflation Indicator)
    cpi_url = f"https://www.alphavantage.co/query?function=INFLATION&apikey={ALPHA_VANTAGE_API_KEY}"
    cpi_response = requests.get(cpi_url)
    macro_indicators['CPI'] = cpi_response.json()
    print("CPI Response:", macro_indicators['CPI'])  # Debug print
    
    # Unemployment Rate
    unemp_url = f"https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey={ALPHA_VANTAGE_API_KEY}"
    unemp_response = requests.get(unemp_url)
    macro_indicators['Unemployment'] = unemp_response.json()
    print("Unemployment Response:", macro_indicators['Unemployment'])  # Debug print
    
    # 10-Year Treasury Rate
    treasury_url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={ALPHA_VANTAGE_API_KEY}"
    treasury_response = requests.get(treasury_url)
    macro_indicators['Treasury Rate'] = treasury_response.json()
    print("Treasury Rate Response:", macro_indicators['Treasury Rate'])  # Debug print
    
    # VIX (Volatility Index)
    vix_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=VIX&apikey={ALPHA_VANTAGE_API_KEY}"
    vix_response = requests.get(vix_url)
    macro_indicators['VIX'] = vix_response.json()
    print("VIX Response:", macro_indicators['VIX'])  # Debug print
    
    return macro_indicators

def plot_macro_indicators(macro_data):
    """Plot all macroeconomic indicators in a 2x3 grid, showing last 20 years of data."""
    plt.figure(figsize=(15, 10))
    
    # Calculate date 20 years ago
    twenty_years_ago = datetime.now() - timedelta(days=365*20)
    
    # GDP
    plt.subplot(2, 3, 1)
    gdp_data = pd.DataFrame(macro_data['GDP']['data']).T
    gdp_data.index = pd.to_datetime(gdp_data.index)
    gdp_data['value'] = pd.to_numeric(gdp_data['value'], errors='coerce')
    gdp_data = gdp_data[gdp_data.index >= twenty_years_ago]
    plt.plot(gdp_data.index, gdp_data['value'])
    plt.title('Real GDP (Last 20 Years)')
    plt.xticks(rotation=45)
    
    # Interest Rates
    plt.subplot(2, 3, 2)
    rates_data = pd.DataFrame(macro_data['Interest Rates']['data']).T
    rates_data.index = pd.to_datetime(rates_data.index)
    rates_data['value'] = pd.to_numeric(rates_data['value'], errors='coerce')
    rates_data = rates_data[rates_data.index >= twenty_years_ago]
    plt.plot(rates_data.index, rates_data['value'])
    plt.title('Federal Funds Rate (Last 20 Years)')
    plt.xticks(rotation=45)
    
    # CPI
    plt.subplot(2, 3, 3)
    cpi_data = pd.DataFrame(macro_data['CPI']['data']).T
    cpi_data.index = pd.to_datetime(cpi_data.index)
    cpi_data['value'] = pd.to_numeric(cpi_data['value'], errors='coerce')
    cpi_data = cpi_data[cpi_data.index >= twenty_years_ago]
    plt.plot(cpi_data.index, cpi_data['value'])
    plt.title('CPI (Inflation) (Last 20 Years)')
    plt.xticks(rotation=45)
    
    # Unemployment
    plt.subplot(2, 3, 4)
    unemp_data = pd.DataFrame(macro_data['Unemployment']['data']).T
    unemp_data.index = pd.to_datetime(unemp_data.index)
    unemp_data['value'] = pd.to_numeric(unemp_data['value'], errors='coerce')
    unemp_data = unemp_data[unemp_data.index >= twenty_years_ago]
    plt.plot(unemp_data.index, unemp_data['value'])
    plt.title('Unemployment Rate (Last 20 Years)')
    plt.xticks(rotation=45)
    
    # Treasury Rate
    plt.subplot(2, 3, 5)
    treasury_data = pd.DataFrame(macro_data['Treasury Rate']['data']).T
    treasury_data.index = pd.to_datetime(treasury_data.index)
    treasury_data['value'] = pd.to_numeric(treasury_data['value'], errors='coerce')
    treasury_data = treasury_data[treasury_data.index >= twenty_years_ago]
    plt.plot(treasury_data.index, treasury_data['value'])
    plt.title('10-Year Treasury Rate (Last 20 Years)')
    plt.xticks(rotation=45)
    
    # VIX
    plt.subplot(2, 3, 6)
    vix_data = pd.DataFrame(macro_data['VIX']['Time Series (Daily)']).T
    vix_data.index = pd.to_datetime(vix_data.index)
    vix_data['4. close'] = pd.to_numeric(vix_data['4. close'], errors='coerce')
    vix_data = vix_data[vix_data.index >= twenty_years_ago]
    plt.plot(vix_data.index, vix_data['4. close'])
    plt.title('VIX (Last 20 Years)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Function to analyze sentiment from news articles
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_analysis(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = " ".join([p.text for p in soup.find_all('p')])
        if not text.strip():
            return 0.0
        sentiment_score = analyzer.polarity_scores(text)["compound"]
        return sentiment_score
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return 0.0

def get_sentiment_analysis_yahoo(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if not text.strip():
            return 0.0
        sentiment_score = analyzer.polarity_scores(text)["compound"]
        return sentiment_score
    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return 0.0

# Example usage
if __name__ == "__main__":
    # Get macroeconomic data and plot it
    macro_data = get_macro_data()
    plot_macro_indicators(macro_data)
    
    # Example sentiment analysis
    news_url = "https://www.bloomberg.com/news/articles/2024-03-01/apple-earnings-report"
    sentiment_score = get_sentiment_analysis(news_url)
    print(f"Sentiment Score: {sentiment_score}")
    
    # Example Yahoo Finance sentiment analysis
    yahoo_url = "https://finance.yahoo.com/news/nvidia-stock-nexthop-ai-arista-stock-ai-stocks/"
    yahoo_sentiment = get_sentiment_analysis_yahoo(yahoo_url)
    print(f"Yahoo Sentiment Score: {yahoo_sentiment}")
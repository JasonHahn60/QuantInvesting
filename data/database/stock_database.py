import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDatabase:
    def __init__(self, db_path: str = "stock_data.db"):
        """Initialize the stock database."""
        self.db_path = db_path
        self.start_date = "2010-01-01"
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create database and tables
        self._init_database()
    
    def _init_database(self):
        """Initialize the database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tickers table with limited precision
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,  -- Keep full precision for market cap
                    pe_ratio REAL,    -- Keep full precision for ratios
                    dividend_yield REAL,
                    last_update TIMESTAMP
                )
            ''')
            
            # Create price_history table with limited precision (2 decimal places)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    ticker TEXT,
                    date DATE,
                    open REAL CHECK(ROUND(open, 2) = open),  -- Enforce 2 decimal places
                    high REAL CHECK(ROUND(high, 2) = high),
                    low REAL CHECK(ROUND(low, 2) = low),
                    close REAL CHECK(ROUND(close, 2) = close),
                    volume INTEGER,
                    dividends REAL CHECK(ROUND(dividends, 2) = dividends),
                    stock_splits REAL CHECK(ROUND(stock_splits, 2) = stock_splits),
                    PRIMARY KEY (ticker, date),
                    FOREIGN KEY (ticker) REFERENCES tickers(ticker)
                )
            ''')
            
            # Create macroeconomic_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS macroeconomic_data (
                    indicator TEXT,
                    date DATE,
                    value REAL CHECK(ROUND(value, 2) = value),  -- Enforce 2 decimal places
                    last_update TIMESTAMP,
                    PRIMARY KEY (indicator, date)
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_history_ticker_date ON price_history(ticker, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_macro_data_indicator_date ON macroeconomic_data(indicator, date)')
            
            conn.commit()
    
    def collect_historical_data(self, tickers: List[str], batch_size: int = 100):
        """Collect historical data for a list of tickers in batches."""
        logger.info(f"Collecting historical data for {len(tickers)} tickers from {self.start_date} to {self.end_date}")
        
        # Process tickers in batches to avoid memory issues
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            logger.info(f"\nProcessing batch {i//batch_size + 1} of {(len(tickers)-1)//batch_size + 1}")
            
            for ticker in batch:
                try:
                    if not self._ticker_exists(ticker):
                        logger.info(f"Fetching data for {ticker}...")
                        stock = yf.Ticker(ticker)
                        
                        # Get historical data
                        try:
                            hist = stock.history(start=self.start_date, end=self.end_date)
                        except Exception as e:
                            logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
                            continue
                        
                        if not hist.empty:
                            try:
                                # Store price data
                                self._store_price_data(ticker, hist)
                                
                                # Store metadata
                                info = stock.info
                                self._store_ticker_metadata(ticker, info)
                                
                                logger.info(f"Successfully collected data for {ticker}")
                            except Exception as e:
                                logger.error(f"Error storing data for {ticker}: {str(e)}")
                        else:
                            logger.warning(f"No data available for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
                    continue
    
    def _ticker_exists(self, ticker: str) -> bool:
        """Check if ticker exists in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM tickers WHERE ticker = ?", (ticker,))
            return cursor.fetchone() is not None
    
    def _store_price_data(self, ticker: str, data: pd.DataFrame):
        """Store price data in the database with limited precision."""
        with sqlite3.connect(self.db_path) as conn:
            # Convert DataFrame to list of tuples with rounded values
            records = [
                (
                    ticker,
                    date.strftime('%Y-%m-%d'),
                    round(row['Open'], 2),
                    round(row['High'], 2),
                    round(row['Low'], 2),
                    round(row['Close'], 2),
                    row['Volume'],
                    round(row['Dividends'], 2),
                    round(row['Stock Splits'], 2)
                )
                for date, row in data.iterrows()
            ]
            
            # Insert data
            conn.executemany('''
                INSERT OR REPLACE INTO price_history 
                (ticker, date, open, high, low, close, volume, dividends, stock_splits)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
            
            conn.commit()
    
    def _store_ticker_metadata(self, ticker: str, info: dict):
        """Store ticker metadata in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO tickers 
                (ticker, name, sector, industry, market_cap, pe_ratio, dividend_yield, last_update)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                info.get('longName', ''),
                info.get('sector', ''),
                info.get('industry', ''),
                info.get('marketCap', None),
                info.get('trailingPE', None),
                info.get('dividendYield', None),
                datetime.now()
            ))
            
            conn.commit()
    
    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Get historical data for a specific ticker."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT date, open, high, low, close, volume, dividends, stock_splits
                FROM price_history
                WHERE ticker = ?
                ORDER BY date
            '''
            df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=['date'])
            df.set_index('date', inplace=True)
            return df
    
    def get_ticker_metadata(self, ticker: str) -> dict:
        """Get metadata for a specific ticker."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tickers WHERE ticker = ?", (ticker,))
            row = cursor.fetchone()
            if row:
                return {
                    'ticker': row[0],
                    'name': row[1],
                    'sector': row[2],
                    'industry': row[3],
                    'market_cap': row[4],
                    'pe_ratio': row[5],
                    'dividend_yield': row[6],
                    'last_update': row[7]
                }
            return {}

    def store_macroeconomic_data(self, macro_data: dict):
        """Store macroeconomic data in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for indicator, data in macro_data.items():
                if 'data' in data:  # For indicators with data array
                    for date, value in data['data'].items():
                        try:
                            # Convert value to float and round to 2 decimal places
                            numeric_value = round(float(value), 2)
                            cursor.execute('''
                                INSERT OR REPLACE INTO macroeconomic_data 
                                (indicator, date, value, last_update)
                                VALUES (?, ?, ?, ?)
                            ''', (indicator, date, numeric_value, datetime.now()))
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert value for {indicator} on {date}")
                elif 'Time Series (Daily)' in data:  # For VIX data
                    for date, values in data['Time Series (Daily)'].items():
                        try:
                            # Get the closing value and round to 2 decimal places
                            numeric_value = round(float(values['4. close']), 2)
                            cursor.execute('''
                                INSERT OR REPLACE INTO macroeconomic_data 
                                (indicator, date, value, last_update)
                                VALUES (?, ?, ?, ?)
                            ''', (indicator, date, numeric_value, datetime.now()))
                        except (ValueError, TypeError, KeyError):
                            logger.warning(f"Could not convert value for {indicator} on {date}")
            
            conn.commit()
            logger.info("Successfully stored macroeconomic data")

    def get_macroeconomic_data(self, indicator: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get macroeconomic data from the database."""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT indicator, date, value
                FROM macroeconomic_data
            '''
            params = []
            
            if indicator or start_date or end_date:
                query += ' WHERE '
                conditions = []
                
                if indicator:
                    conditions.append('indicator = ?')
                    params.append(indicator)
                
                if start_date:
                    conditions.append('date >= ?')
                    params.append(start_date)
                
                if end_date:
                    conditions.append('date <= ?')
                    params.append(end_date)
                
                query += ' AND '.join(conditions)
            
            query += ' ORDER BY date'
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
            return df

# Example usage
if __name__ == "__main__":
    # Initialize database
    db = StockDatabase()
    
    # Get list of tickers and limit to 100 for testing
    from stock_tickers import get_all_tickers
    all_tickers = get_all_tickers()[:100]  # Limit to first 100 tickers
    print(f"\nTesting with {len(all_tickers)} tickers")
    
    # Collect data for all 100 tickers
    db.collect_historical_data(all_tickers)
    
    # Example: Get data for a specific ticker
    ticker = "AAPL"
    data = db.get_ticker_data(ticker)
    metadata = db.get_ticker_metadata(ticker)
    
    print(f"\nData for {ticker}:")
    if not data.empty:
        print(f"Number of trading days: {len(data)}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
    else:
        print("No data available for this ticker yet. Please collect data first.")
    
    print("\nMetadata:")
    if metadata:
        print(metadata)
    else:
        print("No metadata available for this ticker yet. Please collect data first.")
    
    # Example: Store and retrieve macroeconomic data
    from Data_Collection import get_macro_data
    macro_data = get_macro_data()
    db.store_macroeconomic_data(macro_data)
    
    # Get all macroeconomic data
    macro_df = db.get_macroeconomic_data()
    print("\nMacroeconomic Data:")
    print(macro_df.head())
    
    # Get specific indicator data
    gdp_data = db.get_macroeconomic_data(indicator='GDP')
    print("\nGDP Data:")
    print(gdp_data.head()) 
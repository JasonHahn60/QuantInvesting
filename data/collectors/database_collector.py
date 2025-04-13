"""
Database collector for storing and retrieving stock data.
"""
import sqlite3
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseCollector:
    """Manages stock data storage and retrieval in SQLite database."""
    
    def __init__(self, db_path: str = "data/stock_data.db"):
        """Initialize database connection and tables."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Create tickers table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    last_updated TIMESTAMP
                )
            """)
            
            # Create price history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adjusted_close REAL,
                    PRIMARY KEY (symbol, date),
                    FOREIGN KEY (symbol) REFERENCES tickers(symbol)
                )
            """)
            
            # Create sentiment table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    symbol TEXT,
                    date DATE,
                    sentiment_score REAL,
                    sentiment_magnitude REAL,
                    source TEXT,
                    article_count INTEGER,
                    PRIMARY KEY (symbol, date, source),
                    FOREIGN KEY (symbol) REFERENCES tickers(symbol)
                )
            """)
            
            # Create macro data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    date DATE PRIMARY KEY,
                    gdp REAL,
                    interest_rate REAL,
                    cpi REAL,
                    unemployment REAL,
                    treasury_rate REAL,
                    vix REAL
                )
            """)
            
            # Create indices for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_history(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_data(date)")
    
    def store_ticker_data(self, ticker_data: Dict):
        """Store or update ticker information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tickers 
                (symbol, name, sector, industry, market_cap, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker_data['symbol'],
                ticker_data.get('name'),
                ticker_data.get('sector'),
                ticker_data.get('industry'),
                ticker_data.get('market_cap'),
                datetime.now()
            ))
    
    def store_price_data(self, symbol: str, price_data: pd.DataFrame):
        """Store historical price data for a symbol."""
        if price_data.empty:
            return
            
        price_data = price_data.copy()
        price_data['symbol'] = symbol
        
        with sqlite3.connect(self.db_path) as conn:
            price_data.to_sql('price_history', conn, if_exists='append', index=False)
    
    def store_sentiment_data(self, sentiment_data: pd.DataFrame):
        """Store sentiment analysis data."""
        if sentiment_data.empty:
            return
            
        with sqlite3.connect(self.db_path) as conn:
            sentiment_data.to_sql('sentiment_data', conn, if_exists='append', index=False)
    
    def store_macro_data(self, macro_data: Dict):
        """Store macroeconomic data."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO macro_data 
                (date, gdp, interest_rate, cpi, unemployment, treasury_rate, vix)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().date(),
                macro_data.get('GDP'),
                macro_data.get('Interest Rates'),
                macro_data.get('CPI'),
                macro_data.get('Unemployment'),
                macro_data.get('Treasury Rate'),
                macro_data.get('VIX')
            ))
    
    def get_price_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve historical price data for a symbol."""
        query = "SELECT * FROM price_history WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_sentiment_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve sentiment data for a symbol."""
        query = "SELECT * FROM sentiment_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_macro_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve macroeconomic data."""
        query = "SELECT * FROM macro_data"
        params = []
        
        if start_date:
            query += " WHERE date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?" if start_date else " WHERE date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_all_tickers(self) -> pd.DataFrame:
        """Retrieve all ticker information."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM tickers", conn)

# Example usage
if __name__ == "__main__":
    collector = DatabaseCollector()
    print("Database initialized successfully") 
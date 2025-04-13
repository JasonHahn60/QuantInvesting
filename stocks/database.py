"""
Database operations for stock data.
"""
import sqlite3
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class StockDatabase:
    """Manages stock data storage and retrieval."""
    
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
            
            # Create indices for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price_date ON price_history(date)")
    
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
    
    def get_ticker_data(self, symbol: str) -> Dict:
        """Retrieve ticker information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tickers WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            if row:
                return {
                    'symbol': row[0],
                    'name': row[1],
                    'sector': row[2],
                    'industry': row[3],
                    'market_cap': row[4],
                    'last_updated': row[5]
                }
            return {}
    
    def get_all_tickers(self) -> List[str]:
        """Retrieve all ticker symbols."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol FROM tickers")
            return [row[0] for row in cursor.fetchall()] 
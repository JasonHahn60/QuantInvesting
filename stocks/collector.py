"""
Stock data collector using yfinance.
"""
import yfinance as yf
import pandas as pd
import logging
from typing import List, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StockCollector:
    """Collects stock data using yfinance."""
    
    def __init__(self):
        """Initialize the stock data collector."""
        pass
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical stock data for a symbol."""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get stock information and metadata."""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                'symbol': symbol,
                'name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'last_updated': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple stocks."""
        data = {}
        for symbol in symbols:
            df = self.get_stock_data(symbol, period)
            if not df.empty:
                data[symbol] = df
        return data 
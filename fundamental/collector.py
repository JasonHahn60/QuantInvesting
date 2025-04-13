"""
Fundamental data collector using Alpha Vantage API.
"""
import requests
import logging
from typing import Dict
from utils.config import ALPHA_VANTAGE_API_KEY

logger = logging.getLogger(__name__)

class FundamentalCollector:
    """Collects fundamental data from Alpha Vantage."""
    
    def __init__(self):
        """Initialize the fundamental data collector."""
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_income_statement(self, symbol: str) -> Dict:
        """Get income statement data."""
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_balance_sheet(self, symbol: str) -> Dict:
        """Get balance sheet data."""
        params = {
            "function": "BALANCE_SHEET",
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_cash_flow(self, symbol: str) -> Dict:
        """Get cash flow data."""
        params = {
            "function": "CASH_FLOW",
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_earnings(self, symbol: str) -> Dict:
        """Get earnings data."""
        params = {
            "function": "EARNINGS",
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_overview(self, symbol: str) -> Dict:
        """Get company overview."""
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def collect_all(self, symbol: str) -> Dict:
        """Collect all fundamental data for a symbol."""
        logger.info(f"Collecting fundamental data for {symbol}...")
        
        fundamental_data = {
            'income_statement': self.get_income_statement(symbol),
            'balance_sheet': self.get_balance_sheet(symbol),
            'cash_flow': self.get_cash_flow(symbol),
            'earnings': self.get_earnings(symbol),
            'overview': self.get_overview(symbol)
        }
        
        logger.info(f"Successfully collected fundamental data for {symbol}")
        return fundamental_data 
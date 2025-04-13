"""
Macroeconomic data collector using Alpha Vantage API.
"""
import requests
import logging
from datetime import datetime
from typing import Dict
from utils.config import ALPHA_VANTAGE_API_KEY, MACRO_DECIMAL_PLACES

logger = logging.getLogger(__name__)

class MacroCollector:
    """Collects macroeconomic data from Alpha Vantage."""
    
    def __init__(self):
        """Initialize the macroeconomic data collector."""
        self.api_key = ALPHA_VANTAGE_API_KEY
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_gdp(self) -> Dict:
        """Get Real GDP data."""
        params = {
            "function": "REAL_GDP",
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_interest_rates(self) -> Dict:
        """Get Federal Funds Rate data."""
        params = {
            "function": "FEDERAL_FUNDS_RATE",
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_cpi(self) -> Dict:
        """Get CPI (Inflation) data."""
        params = {
            "function": "INFLATION",
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_unemployment(self) -> Dict:
        """Get Unemployment Rate data."""
        params = {
            "function": "UNEMPLOYMENT",
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_treasury_rate(self) -> Dict:
        """Get 10-Year Treasury Rate data."""
        params = {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": "10year",
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def get_vix(self) -> Dict:
        """Get VIX (Volatility Index) data."""
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": "VIX",
            "apikey": self.api_key
        }
        response = requests.get(self.base_url, params=params)
        return response.json()
    
    def collect_all(self) -> Dict:
        """Collect all macroeconomic indicators."""
        logger.info("Collecting macroeconomic data...")
        
        macro_data = {
            'GDP': self.get_gdp(),
            'Interest Rates': self.get_interest_rates(),
            'CPI': self.get_cpi(),
            'Unemployment': self.get_unemployment(),
            'Treasury Rate': self.get_treasury_rate(),
            'VIX': self.get_vix()
        }
        
        logger.info("Successfully collected macroeconomic data")
        return macro_data

# Example usage
if __name__ == "__main__":
    collector = MacroCollector()
    macro_data = collector.collect_all()
    print("Macroeconomic data collected successfully") 
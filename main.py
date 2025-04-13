"""
Main entry point for the quantitative analysis system.
"""
import logging
from datetime import datetime, timedelta
from typing import List

from macro.collector import MacroCollector
from macro.database import MacroDatabase
from stocks.collector import StockCollector
from stocks.database import StockDatabase
from fundamental.collector import FundamentalCollector
from fundamental.database import FundamentalDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_macro_data():
    """Collect and store macroeconomic data."""
    logger.info("Collecting macroeconomic data...")
    collector = MacroCollector()
    database = MacroDatabase()
    
    macro_data = collector.collect_all()
    database.store_macro_data(macro_data)
    
    logger.info("Macroeconomic data collection complete")

def collect_stock_data(symbols: List[str]):
    """Collect and store stock data."""
    logger.info("Collecting stock data...")
    collector = StockCollector()
    database = StockDatabase()
    
    for symbol in symbols:
        # Get stock info
        stock_info = collector.get_stock_info(symbol)
        if stock_info:
            database.store_ticker_data(stock_info)
        
        # Get price data
        price_data = collector.get_stock_data(symbol)
        if not price_data.empty:
            database.store_price_data(symbol, price_data)
    
    logger.info("Stock data collection complete")

def collect_fundamental_data(symbols: List[str]):
    """Collect and store fundamental data."""
    logger.info("Collecting fundamental data...")
    collector = FundamentalCollector()
    database = FundamentalDatabase()
    
    for symbol in symbols:
        fundamental_data = collector.collect_all(symbol)
        database.store_fundamental_data(symbol, fundamental_data)
    
    logger.info("Fundamental data collection complete")

def main():
    """Main entry point."""
    try:
        # Get list of symbols to track
        stock_db = StockDatabase()
        symbols = stock_db.get_all_tickers()
        
        if not symbols:
            logger.warning("No symbols found in database. Please add symbols first.")
            return
        
        # Collect data
        collect_macro_data()
        collect_stock_data(symbols)
        collect_fundamental_data(symbols)
        
        logger.info("Data collection complete")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
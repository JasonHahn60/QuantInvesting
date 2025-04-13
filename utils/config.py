"""
Configuration settings for the quant project.
"""

# API Keys
ALPHA_VANTAGE_API_KEY = "9RBLWH3XU1NBWORU"

# Database settings
DATABASE_PATH = "data/stock_data.db"

# Data collection settings
START_DATE = "2010-01-01"
END_DATE = None  # Will use current date if None

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Data precision settings
PRICE_DECIMAL_PLACES = 2
MACRO_DECIMAL_PLACES = 2 
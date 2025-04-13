"""
Helper functions for the quant project.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List
import pandas as pd

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def safe_float_conversion(value: Any, decimal_places: int = 2) -> float:
    """Safely convert a value to float with specified decimal places."""
    try:
        return round(float(value), decimal_places)
    except (ValueError, TypeError):
        return 0.0

def format_date(date: str) -> str:
    """Format date string to YYYY-MM-DD format."""
    try:
        return datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        return date

def merge_dataframes(dfs: List[pd.DataFrame], on: str = 'date') -> pd.DataFrame:
    """Merge multiple dataframes on a common column."""
    if not dfs:
        return pd.DataFrame()
    
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, on=on, how='outer')
    
    return result

def calculate_returns(df: pd.DataFrame, column: str = 'close') -> pd.Series:
    """Calculate percentage returns for a given column."""
    return df[column].pct_change() * 100

def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Calculate rolling volatility for a given window."""
    returns = calculate_returns(df)
    return returns.rolling(window=window).std()

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data to have zero mean and unit variance."""
    return (df - df.mean()) / df.std()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by removing NaN values and duplicates."""
    return df.dropna().drop_duplicates() 
"""
Database handler for macroeconomic data.
"""
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Optional
import pandas as pd
from utils.config import DATABASE_PATH, MACRO_DECIMAL_PLACES

logger = logging.getLogger(__name__)

class MacroDatabase:
    """Handles storage and retrieval of macroeconomic data."""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        """Initialize the macroeconomic database handler."""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create macroeconomic_data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS macroeconomic_data (
                    indicator TEXT,
                    date DATE,
                    value REAL CHECK(ROUND(value, ?) = value),  -- Enforce decimal places
                    last_update TIMESTAMP,
                    PRIMARY KEY (indicator, date)
                )
            ''', (MACRO_DECIMAL_PLACES,))
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_macro_data_indicator_date 
                ON macroeconomic_data(indicator, date)
            ''')
            
            conn.commit()
    
    def store_macroeconomic_data(self, macro_data: Dict):
        """Store macroeconomic data in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for indicator, data in macro_data.items():
                if 'data' in data:  # For indicators with data array
                    for date, value in data['data'].items():
                        try:
                            # Convert value to float and round to specified decimal places
                            numeric_value = round(float(value), MACRO_DECIMAL_PLACES)
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
                            # Get the closing value and round to specified decimal places
                            numeric_value = round(float(values['4. close']), MACRO_DECIMAL_PLACES)
                            cursor.execute('''
                                INSERT OR REPLACE INTO macroeconomic_data 
                                (indicator, date, value, last_update)
                                VALUES (?, ?, ?, ?)
                            ''', (indicator, date, numeric_value, datetime.now()))
                        except (ValueError, TypeError, KeyError):
                            logger.warning(f"Could not convert value for {indicator} on {date}")
            
            conn.commit()
            logger.info("Successfully stored macroeconomic data")
    
    def get_macroeconomic_data(
        self, 
        indicator: Optional[str] = None, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
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
    from data.collectors.macro_collector import MacroCollector
    
    # Initialize collector and database
    collector = MacroCollector()
    db = MacroDatabase()
    
    # Collect and store data
    macro_data = collector.collect_all()
    db.store_macroeconomic_data(macro_data)
    
    # Retrieve and display data
    all_data = db.get_macroeconomic_data()
    print("\nAll Macroeconomic Data:")
    print(all_data.head())
    
    gdp_data = db.get_macroeconomic_data(indicator='GDP')
    print("\nGDP Data:")
    print(gdp_data.head()) 
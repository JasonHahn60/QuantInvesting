"""
Database operations for macroeconomic data.
"""
import sqlite3
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class MacroDatabase:
    """Manages macroeconomic data storage and retrieval."""
    
    def __init__(self, db_path: str = "data/macro_data.db"):
        """Initialize database connection and tables."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_data(date)")
    
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
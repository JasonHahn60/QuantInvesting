"""
Database operations for fundamental data.
"""
import sqlite3
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FundamentalDatabase:
    """Manages fundamental data storage and retrieval."""
    
    def __init__(self, db_path: str = "data/fundamental_data.db"):
        """Initialize database connection and tables."""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Create income statement table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS income_statement (
                    symbol TEXT,
                    date DATE,
                    revenue REAL,
                    gross_profit REAL,
                    operating_income REAL,
                    net_income REAL,
                    eps REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Create balance sheet table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS balance_sheet (
                    symbol TEXT,
                    date DATE,
                    total_assets REAL,
                    total_liabilities REAL,
                    total_equity REAL,
                    cash REAL,
                    debt REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Create cash flow table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cash_flow (
                    symbol TEXT,
                    date DATE,
                    operating_cash_flow REAL,
                    investing_cash_flow REAL,
                    financing_cash_flow REAL,
                    free_cash_flow REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Create earnings table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS earnings (
                    symbol TEXT,
                    date DATE,
                    reported_eps REAL,
                    estimated_eps REAL,
                    surprise REAL,
                    surprise_percentage REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Create overview table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS overview (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    pe_ratio REAL,
                    pb_ratio REAL,
                    dividend_yield REAL,
                    last_updated TIMESTAMP
                )
            """)
            
            # Create indices for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_income_date ON income_statement(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_balance_date ON balance_sheet(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cash_flow_date ON cash_flow(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings(date)")
    
    def store_fundamental_data(self, symbol: str, fundamental_data: Dict):
        """Store fundamental data for a symbol."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store income statement
            if 'income_statement' in fundamental_data:
                for item in fundamental_data['income_statement'].get('annualReports', []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO income_statement 
                        (symbol, date, revenue, gross_profit, operating_income, net_income, eps)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        item.get('fiscalDateEnding'),
                        item.get('totalRevenue'),
                        item.get('grossProfit'),
                        item.get('operatingIncome'),
                        item.get('netIncome'),
                        item.get('basicEPS')
                    ))
            
            # Store balance sheet
            if 'balance_sheet' in fundamental_data:
                for item in fundamental_data['balance_sheet'].get('annualReports', []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO balance_sheet 
                        (symbol, date, total_assets, total_liabilities, total_equity, cash, debt)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        item.get('fiscalDateEnding'),
                        item.get('totalAssets'),
                        item.get('totalLiabilities'),
                        item.get('totalShareholderEquity'),
                        item.get('cashAndCashEquivalentsAtCarryingValue'),
                        item.get('shortLongTermDebtTotal')
                    ))
            
            # Store cash flow
            if 'cash_flow' in fundamental_data:
                for item in fundamental_data['cash_flow'].get('annualReports', []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO cash_flow 
                        (symbol, date, operating_cash_flow, investing_cash_flow, financing_cash_flow, free_cash_flow)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        item.get('fiscalDateEnding'),
                        item.get('operatingCashflow'),
                        item.get('cashflowFromInvestment'),
                        item.get('cashflowFromFinancing'),
                        item.get('freeCashFlow')
                    ))
            
            # Store earnings
            if 'earnings' in fundamental_data:
                for item in fundamental_data['earnings'].get('quarterlyEarnings', []):
                    cursor.execute("""
                        INSERT OR REPLACE INTO earnings 
                        (symbol, date, reported_eps, estimated_eps, surprise, surprise_percentage)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        item.get('fiscalDateEnding'),
                        item.get('reportedEPS'),
                        item.get('estimatedEPS'),
                        item.get('surprise'),
                        item.get('surprisePercentage')
                    ))
            
            # Store overview
            if 'overview' in fundamental_data:
                overview = fundamental_data['overview']
                cursor.execute("""
                    INSERT OR REPLACE INTO overview 
                    (symbol, name, description, sector, industry, market_cap, pe_ratio, pb_ratio, dividend_yield, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    overview.get('Name'),
                    overview.get('Description'),
                    overview.get('Sector'),
                    overview.get('Industry'),
                    overview.get('MarketCapitalization'),
                    overview.get('PERatio'),
                    overview.get('PriceToBookRatio'),
                    overview.get('DividendYield'),
                    datetime.now()
                ))
    
    def get_fundamental_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        """Retrieve fundamental data for a symbol."""
        data = {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Get income statement
            query = "SELECT * FROM income_statement WHERE symbol = ?"
            params = [symbol]
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            query += " ORDER BY date"
            data['income_statement'] = pd.read_sql_query(query, conn, params=params)
            
            # Get balance sheet
            query = "SELECT * FROM balance_sheet WHERE symbol = ?"
            params = [symbol]
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            query += " ORDER BY date"
            data['balance_sheet'] = pd.read_sql_query(query, conn, params=params)
            
            # Get cash flow
            query = "SELECT * FROM cash_flow WHERE symbol = ?"
            params = [symbol]
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            query += " ORDER BY date"
            data['cash_flow'] = pd.read_sql_query(query, conn, params=params)
            
            # Get earnings
            query = "SELECT * FROM earnings WHERE symbol = ?"
            params = [symbol]
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            query += " ORDER BY date"
            data['earnings'] = pd.read_sql_query(query, conn, params=params)
            
            # Get overview
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM overview WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            if row:
                data['overview'] = {
                    'symbol': row[0],
                    'name': row[1],
                    'description': row[2],
                    'sector': row[3],
                    'industry': row[4],
                    'market_cap': row[5],
                    'pe_ratio': row[6],
                    'pb_ratio': row[7],
                    'dividend_yield': row[8],
                    'last_updated': row[9]
                }
        
        return data 
# src/utils/database.py
import sqlite3
import pandas as pd
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from .config import get_config
from .logger import get_logger

logger = get_logger("database")

class DatabaseManager:
    """Centralized database management"""
    
    def __init__(self, db_path: Optional[str] = None):
        config = get_config()
        if db_path is None:
            db_dir = config.database_dir
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, 'stock_data.db')
        
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database with proper settings"""
        with self.get_connection() as conn:
            # Enable WAL mode for better concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            # Set timeout for busy database
            conn.execute("PRAGMA busy_timeout=30000")
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            return cursor.fetchone() is not None
    
    def get_tables(self) -> List[str]:
        """Get list of all tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [row[0] for row in cursor.fetchall()]
    
    def drop_table(self, table_name: str) -> bool:
        """Drop table if exists"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.commit()
                logger.info(f"Dropped table: {table_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {e}")
            return False
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, 
                      if_exists: str = 'replace') -> bool:
        """Save DataFrame to database"""
        try:
            with self.get_connection() as conn:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
                logger.info(f"Saved {len(df)} rows to table: {table_name}")
                return True
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {table_name}: {e}")
            return False
    
    def load_dataframe(self, table_name: str, query: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load DataFrame from database"""
        try:
            with self.get_connection() as conn:
                if query is None:
                    query = f"SELECT * FROM {table_name}"
                df = pd.read_sql(query, conn)
                logger.info(f"Loaded {len(df)} rows from table: {table_name}")
                return df
        except Exception as e:
            logger.error(f"Failed to load DataFrame from {table_name}: {e}")
            return None
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get table information"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                
                return {
                    'row_count': row_count,
                    'columns': columns,
                    'table_name': table_name
                }
        except Exception as e:
            logger.error(f"Failed to get info for table {table_name}: {e}")
            return {}
    
    def clear_ticker_tables(self, tickers: List[str]):
        """Clear all tables for given tickers"""
        tables_to_drop = []
        
        # Add ticker-specific tables
        for ticker in tickers:
            tables_to_drop.extend([
                f"{ticker}_OHLCV",
                f"{ticker}_Fundamental"
            ])
        
        # Add common tables
        tables_to_drop.extend(['VNINDEX', 'All_OHLCV'])
        
        for table in tables_to_drop:
            self.drop_table(table)

# Global database instance
_db_instance = None

def get_database(db_path: Optional[str] = None) -> DatabaseManager:
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(db_path)
    return _db_instance
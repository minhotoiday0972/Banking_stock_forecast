# src/utils/data_viewer.py
import pandas as pd
import argparse
from typing import Optional, List

from .config import get_config
from .logger import get_logger
from .database import get_database

logger = get_logger("data_viewer")

class DataViewer:
    """Utility for viewing database contents"""
    
    def __init__(self):
        self.config = get_config()
        self.db = get_database()
    
    def view_all_tables(self):
        """View all tables in database"""
        tables = self.db.get_tables()
        
        if not tables:
            print("⚠️ No tables found in the database")
            return
        
        print(f"\n📋 Found {len(tables)} tables in the database:")
        print("=" * 60)
        
        for table in tables:
            info = self.db.get_table_info(table)
            if info:
                print(f"\n📊 Table: {table}")
                print(f"   - Rows: {info['row_count']:,}")
                print(f"   - Columns: {', '.join(info['columns'])}")
                
                # Show sample data
                sample_df = self.db.load_dataframe(table, f"SELECT * FROM {table} LIMIT 3")
                if sample_df is not None and not sample_df.empty:
                    print("   - Sample data:")
                    print(sample_df.to_string(index=False, max_cols=6))
            else:
                print(f"❌ Could not get info for table: {table}")
            
            print("-" * 60)
    
    def view_table(self, table_name: str, limit: int = 10):
        """View specific table"""
        if not self.db.table_exists(table_name):
            print(f"❌ Table '{table_name}' does not exist")
            return
        
        info = self.db.get_table_info(table_name)
        if not info:
            print(f"❌ Could not get info for table: {table_name}")
            return
        
        print(f"\n📊 Table: {table_name}")
        print(f"   - Rows: {info['row_count']:,}")
        print(f"   - Columns: {', '.join(info['columns'])}")
        
        # Load data
        query = f"SELECT * FROM {table_name} ORDER BY time DESC LIMIT {limit}" if 'time' in info['columns'] else f"SELECT * FROM {table_name} LIMIT {limit}"
        df = self.db.load_dataframe(table_name, query)
        
        if df is not None and not df.empty:
            print(f"\n📋 Data (showing {len(df)} rows):")
            print(df.to_string(index=False))
        else:
            print("⚠️ No data found")
    
    def view_ticker_data(self, ticker: str):
        """View all data for a specific ticker"""
        print(f"\n🏦 Data for ticker: {ticker}")
        print("=" * 60)
        
        # OHLCV data
        ohlcv_table = f"{ticker}_OHLCV"
        if self.db.table_exists(ohlcv_table):
            print(f"\n📈 OHLCV Data:")
            self.view_table(ohlcv_table, 5)
        else:
            print(f"❌ No OHLCV data found for {ticker}")
        
        # Fundamental data
        fundamental_table = f"{ticker}_Fundamental"
        if self.db.table_exists(fundamental_table):
            print(f"\n📊 Fundamental Data:")
            self.view_table(fundamental_table, 5)
        else:
            print(f"❌ No fundamental data found for {ticker}")
        
        # Features data
        features_path = f"{self.config.processed_dir}/{ticker}_features.csv"
        try:
            features_df = pd.read_csv(features_path)
            print(f"\n🔧 Processed Features:")
            print(f"   - Rows: {len(features_df):,}")
            print(f"   - Columns: {len(features_df.columns)}")
            print(f"   - Feature columns: {', '.join(features_df.columns[:10])}{'...' if len(features_df.columns) > 10 else ''}")
            print(f"\n📋 Latest features (last 3 rows):")
            print(features_df.tail(3).to_string(index=False, max_cols=8))
        except FileNotFoundError:
            print(f"❌ No processed features found for {ticker}")
        except Exception as e:
            print(f"❌ Error loading features for {ticker}: {e}")
    
    def view_market_data(self):
        """View market-wide data"""
        print(f"\n🌐 Market Data")
        print("=" * 60)
        
        # VN-Index
        if self.db.table_exists('VNINDEX'):
            print(f"\n📊 VN-Index:")
            self.view_table('VNINDEX', 10)
        else:
            print("❌ No VN-Index data found")
        
        # Combined OHLCV
        if self.db.table_exists('All_OHLCV'):
            info = self.db.get_table_info('All_OHLCV')
            print(f"\n📈 Combined OHLCV:")
            print(f"   - Total rows: {info['row_count']:,}")
            
            # Show ticker distribution
            query = "SELECT Ticker, COUNT(*) as count FROM All_OHLCV GROUP BY Ticker ORDER BY count DESC"
            ticker_counts = self.db.load_dataframe('All_OHLCV', query)
            if ticker_counts is not None:
                print(f"   - Ticker distribution:")
                print(ticker_counts.to_string(index=False))
        else:
            print("❌ No combined OHLCV data found")
    
    def check_data_quality(self, ticker: Optional[str] = None):
        """Check data quality issues"""
        print(f"\n🔍 Data Quality Check")
        print("=" * 60)
        
        if ticker:
            tickers = [ticker]
        else:
            tickers = self.config.tickers
        
        for ticker in tickers:
            print(f"\n🏦 Checking {ticker}:")
            
            # Check OHLCV data
            ohlcv_table = f"{ticker}_OHLCV"
            if self.db.table_exists(ohlcv_table):
                df = self.db.load_dataframe(ohlcv_table)
                if df is not None:
                    # Check for missing values
                    missing = df.isnull().sum()
                    if missing.sum() > 0:
                        print(f"   ⚠️ OHLCV missing values: {missing[missing > 0].to_dict()}")
                    
                    # Check for invalid prices
                    invalid_prices = (df[['Open', 'High', 'Low', 'Close']] <= 0).sum()
                    if invalid_prices.sum() > 0:
                        print(f"   ⚠️ Invalid prices: {invalid_prices[invalid_prices > 0].to_dict()}")
                    
                    # Check for invalid volumes
                    invalid_volume = (df['Volume'] < 0).sum()
                    if invalid_volume > 0:
                        print(f"   ⚠️ Invalid volume: {invalid_volume} rows")
                    
                    # Check date range
                    df['time'] = pd.to_datetime(df['time'])
                    date_range = f"{df['time'].min().date()} to {df['time'].max().date()}"
                    print(f"   ✅ OHLCV date range: {date_range} ({len(df)} rows)")
                else:
                    print(f"   ❌ Could not load OHLCV data")
            else:
                print(f"   ❌ No OHLCV table found")
            
            # Check fundamental data
            fundamental_table = f"{ticker}_Fundamental"
            if self.db.table_exists(fundamental_table):
                df = self.db.load_dataframe(fundamental_table)
                if df is not None:
                    print(f"   ✅ Fundamental data: {len(df)} rows")
                else:
                    print(f"   ❌ Could not load fundamental data")
            else:
                print(f"   ❌ No fundamental table found")

def main():
    """Main function for data viewer CLI"""
    parser = argparse.ArgumentParser(description="View database contents")
    parser.add_argument('command', choices=['tables', 'table', 'ticker', 'market', 'quality'],
                       help="What to view")
    parser.add_argument('--name', help="Table name or ticker symbol")
    parser.add_argument('--limit', type=int, default=10, help="Number of rows to show")
    
    args = parser.parse_args()
    
    viewer = DataViewer()
    
    if args.command == 'tables':
        viewer.view_all_tables()
    elif args.command == 'table':
        if not args.name:
            print("❌ Please specify table name with --name")
            return
        viewer.view_table(args.name, args.limit)
    elif args.command == 'ticker':
        if not args.name:
            print("❌ Please specify ticker with --name")
            return
        viewer.view_ticker_data(args.name)
    elif args.command == 'market':
        viewer.view_market_data()
    elif args.command == 'quality':
        viewer.check_data_quality(args.name)

if __name__ == "__main__":
    main()
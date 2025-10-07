#!/usr/bin/env python3
"""
Script to check available banking-specific data from vnstock
"""
import sys
import os
import pandas as pd

# Add parent directory and src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, parent_dir)
sys.path.insert(0, src_dir)

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.data.data_collector import DataCollector

logger = get_logger("banking_data_checker")

def check_available_columns(ticker: str = "VCB"):
    """Check what columns are available from vnstock for banking data"""
    try:
        collector = DataCollector()
        
        # Get raw data to see all available columns
        ratios = collector.vnstock_obj.stock(symbol=ticker, source='TCBS').finance.ratio(
            period='quarter', lang='vi'
        )
        
        if ratios is None or ratios.empty:
            print(f"❌ No data available for {ticker}")
            return
        
        print(f"📊 Available columns for {ticker}:")
        print("=" * 60)
        
        # Print all columns
        for i, col in enumerate(ratios.columns, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\n📋 Total columns: {len(ratios.columns)}")
        
        # Check for banking-specific columns
        banking_keywords = [
            'npl', 'loan', 'deposit', 'interest', 'margin', 'cost', 'income',
            'casa', 'provision', 'capital', 'adequacy', 'risk'
        ]
        
        print(f"\n🏦 Potential banking-specific columns:")
        print("=" * 60)
        
        banking_cols = []
        for col in ratios.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in banking_keywords):
                banking_cols.append(col)
                print(f"✅ {col}")
        
        if not banking_cols:
            print("❌ No obvious banking-specific columns found")
        
        # Show sample data
        print(f"\n📈 Sample data (last 5 quarters):")
        print("=" * 60)
        sample_cols = ['quarter', 'year'] + banking_cols[:5]  # Show first 5 banking cols
        available_sample_cols = [col for col in sample_cols if col in ratios.columns]
        
        if available_sample_cols:
            print(ratios[available_sample_cols].tail().to_string(index=False))
        else:
            print("No sample data to display")
        
        return ratios.columns.tolist()
        
    except Exception as e:
        logger.error(f"Error checking data for {ticker}: {e}")
        print(f"❌ Error: {e}")
        return None

def check_multiple_banks():
    """Check data availability across multiple banks"""
    config = get_config()
    tickers = config.tickers[:3]  # Check first 3 banks
    
    print("🏦 Checking banking data across multiple banks")
    print("=" * 80)
    
    all_columns = set()
    bank_columns = {}
    
    for ticker in tickers:
        print(f"\n📊 Checking {ticker}...")
        columns = check_available_columns(ticker)
        if columns:
            bank_columns[ticker] = columns
            all_columns.update(columns)
        print("-" * 40)
    
    # Find common columns
    if bank_columns:
        common_columns = set(bank_columns[list(bank_columns.keys())[0]])
        for ticker, columns in bank_columns.items():
            common_columns = common_columns.intersection(set(columns))
        
        print(f"\n🔄 Common columns across all banks ({len(common_columns)}):")
        print("=" * 60)
        for col in sorted(common_columns):
            print(f"✅ {col}")
    
    return bank_columns

def suggest_mapping():
    """Suggest mapping for banking-specific features"""
    print("\n💡 Suggested mapping for banking features:")
    print("=" * 60)
    
    mapping_suggestions = {
        "NPL (%)": ["npl_ratio", "bad_debt_ratio", "non_performing_loan"],
        "NIM (%)": ["net_interest_margin", "nim", "interest_margin"],
        "CIR (%)": ["cost_income_ratio", "cir", "efficiency_ratio"],
        "CASA (%)": ["casa_ratio", "current_savings_ratio", "low_cost_deposit"],
        "Loan_Growth (%)": ["loan_growth", "credit_growth", "lending_growth"],
        "Deposit_Growth (%)": ["deposit_growth", "funding_growth"],
        "Credit_Risk_Provision (%)": ["provision_ratio", "credit_loss_provision"],
        "Capital_Adequacy_Ratio (%)": ["car", "capital_ratio", "tier1_ratio"]
    }
    
    for target, possible_names in mapping_suggestions.items():
        print(f"\n🎯 {target}:")
        for name in possible_names:
            print(f"   - {name}")

def main():
    """Main function"""
    print("🔍 Banking Data Availability Checker")
    print("=" * 80)
    
    # Check single bank first
    print("1️⃣ Checking single bank (VCB)...")
    check_available_columns("VCB")
    
    print("\n" + "=" * 80)
    
    # Check multiple banks
    print("2️⃣ Checking multiple banks...")
    check_multiple_banks()
    
    # Suggest mapping
    suggest_mapping()
    
    print(f"\n✅ Check completed! Review the output to update data collection.")

if __name__ == "__main__":
    main()
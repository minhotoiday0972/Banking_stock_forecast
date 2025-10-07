#!/usr/bin/env python3
"""
Setup script for Vietnamese Banking Stock Predictor
"""
import os
import sys

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/database',
        'models',
        'outputs',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_requirements():
    """Check if requirements are installed"""
    try:
        import pandas
        import numpy
        import torch
        import streamlit
        import vnstock
        import sklearn
        import plotly
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def setup_config():
    """Check config file"""
    if os.path.exists('config.yaml'):
        print("✅ Config file exists")
    else:
        print("❌ Config file not found")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Vietnamese Banking Stock Predictor")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check requirements
    print("\n📦 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Check config
    print("\n⚙️ Checking configuration...")
    if not setup_config():
        sys.exit(1)
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download VN-Index data to data/raw/vnindex_data.csv")
    print("2. Run: python main.py full --models all")
    print("3. Start web app: streamlit run src/app.py")

if __name__ == "__main__":
    main()
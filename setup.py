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
        'logs',
        'mlruns',
        '.cache'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_requirements():
    """Check if requirements are installed"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'torch': 'torch',
        'streamlit': 'streamlit',
        'vnstock': 'vnstock',
        'sklearn': 'scikit-learn',
        'plotly': 'plotly',
        'yaml': 'pyyaml',
        'mlflow': 'mlflow',
        'optuna': 'optuna',
        'ta': 'ta',
        'joblib': 'joblib'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def setup_config():
    """Check config file"""
    if os.path.exists('config.yaml'):
        print("✅ Config file exists")
        return True
    else:
        print("❌ Config file not found")
        return False

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def create_env_file():
    """Create .env file from example if it doesn't exist"""
    if not os.path.exists('.env') and os.path.exists('.env.example'):
        import shutil
        shutil.copy('.env.example', '.env')
        print("✅ Created .env file from .env.example")
    elif os.path.exists('.env'):
        print("✅ .env file exists")
    else:
        print("⚠️  No .env file found (optional)")

def main():
    """Main setup function"""
    print("🚀 Setting up Vietnamese Banking Stock Predictor v2.0")
    print("=" * 60)
    
    # Check Python version
    print("\n🐍 Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check requirements
    print("\n📦 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Check config
    print("\n⚙️  Checking configuration...")
    if not setup_config():
        sys.exit(1)
    
    # Create env file
    print("\n🔧 Setting up environment...")
    create_env_file()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Place VN-Index data in: data/raw/vnindex_data.csv")
    print("2. Run system check: python system_check.py")
    print("3. Check status: python main.py status")
    print("4. Run full pipeline: python main.py full --models all")
    print("5. Start web app: streamlit run app.py")
    print("\n💡 Quick start: make setup && make run-pipeline")

if __name__ == "__main__":
    main()
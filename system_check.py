#!/usr/bin/env python3
"""
System Check - Kiểm tra hệ thống trước khi chạy
"""
import os
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Kiểm tra Python version"""
    print("🐍 Python Version Check:")
    version = sys.version_info
    print(f"   Current: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version OK")
        return True
    else:
        print("   ❌ Python 3.8+ required")
        return False

def check_required_packages():
    """Kiểm tra các package cần thiết"""
    print("\n📦 Required Packages Check:")
    
    required_packages = [
        'torch',
        'pandas', 
        'numpy',
        'scikit-learn',
        'vnstock',
        'streamlit',
        'matplotlib',
        'seaborn',
        'yaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'yaml':
                importlib.import_module('yaml')
            elif package == 'scikit-learn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("   ✅ All packages available")
        return True

def check_project_structure():
    """Kiểm tra cấu trúc project"""
    print("\n📁 Project Structure Check:")
    
    required_files = [
        'config.yaml',
        'main.py',
        'app.py',
        'run_full_pipeline.py',
        'check_results.py',
        'src/utils/config.py',
        'src/utils/database.py',
        'src/utils/logger.py',
        'src/data/data_collector.py',
        'src/features/feature_engineer.py',
        'src/models/base_model.py',
        'src/models/cnn_bilstm.py',
        'src/models/transformer.py',
        'src/training/trainer.py',
        'src/app/predictor.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {len(missing_files)}")
        return False
    else:
        print("   ✅ All core files present")
        return True

def check_directories():
    """Kiểm tra và tạo directories cần thiết"""
    print("\n📂 Directories Check:")
    
    required_dirs = [
        'data/raw',
        'data/processed', 
        'data/database',
        'models',
        'outputs',
        'logs'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   📁 Creating {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    print("   ✅ All directories ready")
    return True

def check_config():
    """Kiểm tra config file"""
    print("\n⚙️ Configuration Check:")
    
    try:
        import yaml
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['data', 'features', 'models', 'training', 'paths']
        for section in required_sections:
            if section in config:
                print(f"   ✅ {section} section")
            else:
                print(f"   ❌ {section} section - MISSING")
                return False
        
        # Check tickers
        tickers = config.get('data', {}).get('tickers', [])
        print(f"   📊 Tickers: {len(tickers)} banks")
        
        # Check models
        models = [k for k in config.get('models', {}).keys() if k != 'forecast_horizons']
        print(f"   🤖 Models: {models}")
        
        print("   ✅ Configuration valid")
        return True
        
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False

def check_gpu():
    """Kiểm tra GPU availability"""
    print("\n🖥️ GPU Check:")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU available: {gpu_name}")
            print(f"   📊 GPU count: {gpu_count}")
            return True
        else:
            print("   ⚠️ No GPU available - will use CPU")
            print("   💡 Training will be slower but still works")
            return True
    except:
        print("   ⚠️ Cannot check GPU status")
        return True

def main():
    """Main system check"""
    print("🔍 SYSTEM CHECK FOR BANKING STOCK PREDICTION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Directories", check_directories),
        ("Configuration", check_config),
        ("GPU", check_gpu)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"   ❌ Error in {check_name}: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SYSTEM CHECK SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {check_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 SYSTEM READY!")
        print("✅ You can start running the project")
        print("\n🚀 Next steps:")
        print("   1. python run_full_pipeline.py  (full pipeline)")
        print("   2. python main.py collect       (data only)")
        print("   3. run_pipeline.bat             (Windows menu)")
    else:
        print("\n⚠️ SYSTEM NOT READY")
        print("❌ Please fix the failed checks before running")
        
        failed_checks = [name for name, result in results.items() if not result]
        print(f"❌ Failed: {failed_checks}")

if __name__ == "__main__":
    main()
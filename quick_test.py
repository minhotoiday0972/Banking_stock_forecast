#!/usr/bin/env python3
"""
Quick Test - Test nhanh với 1 ticker để đảm bảo hệ thống hoạt động
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def test_data_collection():
    """Test data collection với 1 ticker"""
    print("Testing Data Collection...")
    
    from src.data.data_collector import DataCollector
    
    collector = DataCollector()
    
    # Test với VCB only
    available, failed = collector.collect_all_data(tickers=["VCB"])
    
    if available:
        print(f"SUCCESS: Data collected for {available}")
        return True
    else:
        print(f"FAILED: {failed}")
        return False

def test_feature_engineering():
    """Test feature engineering với 1 ticker"""
    print("Testing Feature Engineering...")
    
    from src.features.feature_engineer import FeatureEngineer
    
    engineer = FeatureEngineer()
    results = engineer.process_all_tickers(["VCB"])
    
    if results.get("VCB", False):
        print("SUCCESS: Features engineered for VCB")
        return True
    else:
        print("FAILED: Feature engineering failed")
        return False

def test_model_training():
    """Test model training với 1 ticker, 1 model"""
    print("Testing Model Training...")
    
    from src.training.trainer import ModelTrainingPipeline
    
    pipeline = ModelTrainingPipeline()
    result = pipeline.train_single_model("transformer", "VCB")
    
    if result is not None:
        print("SUCCESS: Model trained for VCB")
        return True
    else:
        print("FAILED: Model training failed")
        return False

def main():
    """Quick test main function"""
    print("QUICK TEST - Vietnamese Banking Stock Prediction")
    print("=" * 60)
    print("Testing with VCB ticker only...")
    
    tests = [
        ("Data Collection", test_data_collection),
        ("Feature Engineering", test_feature_engineering),
        ("Model Training", test_model_training)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"PASS: {test_name}")
            else:
                print(f"FAIL: {test_name}")
                
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("QUICK TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All tests passed!")
        print("System is working correctly.")
        print("You can now run the full pipeline.")
    else:
        print("\nFAILED: Some tests failed.")
        print("Please check the errors above.")

if __name__ == "__main__":
    main()
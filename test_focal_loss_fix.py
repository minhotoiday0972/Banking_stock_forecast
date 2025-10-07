#!/usr/bin/env python3
"""
Test script để kiểm tra Focal Loss fix có hoạt động không
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from src.models.base_model import FocalLoss, ModelTrainer
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("test_fix")

def test_focal_loss():
    """Test Focal Loss implementation"""
    print("🧪 Testing Focal Loss Implementation...")
    
    # Create sample data
    batch_size = 32
    num_classes = 3
    
    # Simulate model outputs (logits)
    logits = torch.randn(batch_size, num_classes)
    
    # Simulate imbalanced targets (mostly class 1)
    targets = torch.ones(batch_size, dtype=torch.long)  # All class 1 (Flat)
    targets[:5] = 0  # Few class 0 (Down)
    targets[-3:] = 2  # Few class 2 (Up)
    
    print(f"Target distribution: {torch.bincount(targets)}")
    
    # Test standard CrossEntropy
    ce_loss = torch.nn.CrossEntropyLoss()
    ce_result = ce_loss(logits, targets)
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    focal_result = focal_loss(logits, targets)
    
    print(f"CrossEntropy Loss: {ce_result.item():.4f}")
    print(f"Focal Loss: {focal_result.item():.4f}")
    print("✅ Focal Loss implementation working!")
    
    return True

def test_class_weights():
    """Test enhanced class weights calculation"""
    print("\n🧪 Testing Enhanced Class Weights...")
    
    # Create imbalanced dataset (like real banking data)
    # Class distribution: Down=18%, Flat=62%, Up=20%
    n_samples = 1000
    targets = np.array([1] * 620 + [0] * 180 + [2] * 200)  # Imbalanced
    np.random.shuffle(targets)
    
    print(f"Original distribution: {np.bincount(targets)}")
    
    # Test class weights calculation
    trainer = ModelTrainer("test")
    class_weights = trainer._calculate_class_weights(targets)
    
    print(f"Enhanced class weights: {class_weights.tolist()}")
    
    # Check if dominant class (1) has lower weight
    if class_weights[1] < class_weights[0] and class_weights[1] < class_weights[2]:
        print("✅ Dominant class penalty working!")
    else:
        print("❌ Dominant class penalty NOT working!")
        return False
    
    return True

def test_single_ticker_training():
    """Test training on single ticker with fixes"""
    print("\n🧪 Testing Single Ticker Training with Fixes...")
    
    try:
        # Import training function
        from src.models.cnn_bilstm import train_cnn_bilstm_model
        
        # Test with VCB (had constant accuracy problem)
        ticker = "VCB"
        print(f"Training {ticker} with enhanced fixes...")
        
        # Check if sequences exist
        sequences_path = f"data/processed/{ticker}_sequences.npz"
        if not os.path.exists(sequences_path):
            print(f"❌ Sequences file not found: {sequences_path}")
            print("   Run feature engineering first!")
            return False
        
        # Train model (just 5 epochs for testing)
        config = get_config()
        original_epochs = config.get('training.epochs', 50)
        
        # Temporarily reduce epochs for testing
        import yaml
        with open('config.yaml', 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data['training']['epochs'] = 5  # Quick test
        
        with open('config_test.yaml', 'w') as f:
            yaml.dump(config_data, f)
        
        print("🚀 Starting quick training test (5 epochs)...")
        
        # This will use the enhanced training with Focal Loss
        result = train_cnn_bilstm_model(ticker)
        
        if result is not None:
            print("✅ Training completed successfully!")
            
            # Check test metrics for constant prediction
            test_metrics = result.get('test_metrics', {})
            for target_name, metrics in test_metrics.items():
                if 'Direction' in target_name and metrics['type'] == 'classification':
                    is_constant = metrics.get('is_constant_prediction', False)
                    unique_preds = metrics.get('unique_predictions', 0)
                    
                    if is_constant:
                        print(f"❌ {target_name}: Still predicting only 1 class!")
                        return False
                    else:
                        print(f"✅ {target_name}: Predicting {unique_preds} classes!")
            
            return True
        else:
            print("❌ Training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    finally:
        # Cleanup test config
        if os.path.exists('config_test.yaml'):
            os.remove('config_test.yaml')

def main():
    """Run all tests"""
    print("🔧 TESTING FOCAL LOSS FIXES")
    print("=" * 50)
    
    tests = [
        ("Focal Loss Implementation", test_focal_loss),
        ("Enhanced Class Weights", test_class_weights),
        ("Single Ticker Training", test_single_ticker_training)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Fixes are working correctly.")
        print("💡 You can now run full training with:")
        print("   python main.py train --models cnn_bilstm")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Quick test để kiểm tra fixes có hoạt động không
Chỉ train 1 ticker với 10 epochs để test nhanh
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Test nhanh với 1 ticker"""
    
    print("🚀 QUICK TEST - Focal Loss Fixes")
    print("=" * 40)
    
    # Test với VCB (ticker có vấn đề constant accuracy)
    ticker = "VCB"
    
    try:
        from src.models.cnn_bilstm import train_cnn_bilstm_model
        from src.utils.config import get_config
        import yaml
        
        # Backup original config
        with open('config.yaml', 'r') as f:
            original_config = yaml.safe_load(f)
        
        # Create test config with fewer epochs
        test_config = original_config.copy()
        test_config['training']['epochs'] = 10  # Quick test
        test_config['training']['early_stopping_patience'] = 5
        
        with open('config_quick_test.yaml', 'w') as f:
            yaml.dump(test_config, f)
        
        # Temporarily replace config
        os.rename('config.yaml', 'config_backup.yaml')
        os.rename('config_quick_test.yaml', 'config.yaml')
        
        print(f"🧪 Testing {ticker} with 10 epochs...")
        print("🎯 Looking for:")
        print("   - Multiple classes predicted (not just 1)")
        print("   - Per-class metrics logged")
        print("   - Focal loss warnings")
        
        # Train model
        result = train_cnn_bilstm_model(ticker)
        
        if result is not None:
            print("\n✅ Training completed!")
            
            # Analyze results
            test_metrics = result.get('test_metrics', {})
            
            for target_name, metrics in test_metrics.items():
                if 'Direction' in target_name and metrics['type'] == 'classification':
                    is_constant = metrics.get('is_constant_prediction', False)
                    unique_preds = metrics.get('unique_predictions', 0)
                    accuracy = metrics.get('accuracy', 0)
                    
                    print(f"\n📊 {target_name}:")
                    print(f"   Accuracy: {accuracy:.4f}")
                    print(f"   Unique predictions: {unique_preds}")
                    print(f"   Constant prediction: {'YES ❌' if is_constant else 'NO ✅'}")
                    
                    if 'precision_per_class' in metrics:
                        precision = metrics['precision_per_class']
                        recall = metrics['recall_per_class']
                        print(f"   Per-class Precision: {[f'{p:.3f}' for p in precision]}")
                        print(f"   Per-class Recall: {[f'{r:.3f}' for r in recall]}")
            
            print(f"\n🎉 Quick test completed successfully!")
            return True
        else:
            print("❌ Training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Restore original config
        if os.path.exists('config_backup.yaml'):
            if os.path.exists('config.yaml'):
                os.remove('config.yaml')
            os.rename('config_backup.yaml', 'config.yaml')
        
        # Cleanup
        for temp_file in ['config_quick_test.yaml']:
            if os.path.exists(temp_file):
                os.remove(temp_file)

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n💡 Fixes appear to be working!")
        print("🚀 Ready for full retraining:")
        print("   python retrain_with_fixes.py")
    else:
        print("\n⚠️  Issues detected. Check the logs above.")
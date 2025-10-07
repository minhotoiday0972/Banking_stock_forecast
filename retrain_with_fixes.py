#!/usr/bin/env python3
"""
Script để retrain tất cả models với Focal Loss fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.models.cnn_bilstm import train_cnn_bilstm_model
from src.models.transformer import train_transformer_model

logger = get_logger("retrain_fixes")

def retrain_all_models():
    """Retrain all models with Focal Loss fixes"""
    
    print("🔧 RETRAINING ALL MODELS WITH FOCAL LOSS FIXES")
    print("=" * 60)
    
    config = get_config()
    tickers = config.tickers
    
    print(f"📋 Tickers to retrain: {tickers}")
    print(f"🎯 Models: CNN-BiLSTM, Transformer")
    print(f"⚡ Enhanced features:")
    print("   - Focal Loss (alpha=1.0, gamma=2.0)")
    print("   - Enhanced class weights (3x penalty)")
    print("   - Classification loss weight = 2.0")
    print("   - Per-class metrics monitoring")
    print("   - Constant prediction detection")
    
    start_time = time.time()
    
    # Results tracking
    results = {
        'cnn_bilstm': {},
        'transformer': {}
    }
    
    # Train CNN-BiLSTM models
    print(f"\n🚀 TRAINING CNN-BiLSTM MODELS")
    print("-" * 40)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Training CNN-BiLSTM for {ticker}...")
        
        try:
            # Check if sequences exist
            sequences_path = f"data/processed/{ticker}_sequences.npz"
            if not os.path.exists(sequences_path):
                print(f"❌ Sequences not found for {ticker}. Skipping...")
                results['cnn_bilstm'][ticker] = None
                continue
            
            # Train model
            result = train_cnn_bilstm_model(ticker)
            results['cnn_bilstm'][ticker] = result
            
            if result is not None:
                # Check for constant prediction issues
                test_metrics = result.get('test_metrics', {})
                constant_predictions = []
                
                for target_name, metrics in test_metrics.items():
                    if 'Direction' in target_name and metrics['type'] == 'classification':
                        if metrics.get('is_constant_prediction', False):
                            constant_predictions.append(target_name)
                
                if constant_predictions:
                    print(f"⚠️  {ticker}: Still has constant predictions in {constant_predictions}")
                else:
                    print(f"✅ {ticker}: All direction predictions are diverse!")
            else:
                print(f"❌ {ticker}: Training failed!")
                
        except Exception as e:
            print(f"❌ {ticker}: Error - {e}")
            results['cnn_bilstm'][ticker] = None
    
    # Train Transformer models
    print(f"\n🚀 TRAINING TRANSFORMER MODELS")
    print("-" * 40)
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Training Transformer for {ticker}...")
        
        try:
            # Check if sequences exist
            sequences_path = f"data/processed/{ticker}_sequences.npz"
            if not os.path.exists(sequences_path):
                print(f"❌ Sequences not found for {ticker}. Skipping...")
                results['transformer'][ticker] = None
                continue
            
            # Train model
            result = train_transformer_model(ticker)
            results['transformer'][ticker] = result
            
            if result is not None:
                # Check for constant prediction issues
                test_metrics = result.get('test_metrics', {})
                constant_predictions = []
                
                for target_name, metrics in test_metrics.items():
                    if 'Direction' in target_name and metrics['type'] == 'classification':
                        if metrics.get('is_constant_prediction', False):
                            constant_predictions.append(target_name)
                
                if constant_predictions:
                    print(f"⚠️  {ticker}: Still has constant predictions in {constant_predictions}")
                else:
                    print(f"✅ {ticker}: All direction predictions are diverse!")
            else:
                print(f"❌ {ticker}: Training failed!")
                
        except Exception as e:
            print(f"❌ {ticker}: Error - {e}")
            results['transformer'][ticker] = None
    
    # Generate summary report
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n" + "=" * 60)
    print("📊 RETRAINING SUMMARY")
    print("=" * 60)
    
    for model_type in ['cnn_bilstm', 'transformer']:
        model_results = results[model_type]
        successful = [t for t, r in model_results.items() if r is not None]
        failed = [t for t, r in model_results.items() if r is None]
        
        print(f"\n{model_type.upper()}:")
        print(f"  ✅ Successful: {len(successful)}/{len(tickers)} ({len(successful)/len(tickers)*100:.1f}%)")
        print(f"  ❌ Failed: {len(failed)}")
        
        if successful:
            print(f"  🎯 Successful tickers: {successful}")
        if failed:
            print(f"  💥 Failed tickers: {failed}")
    
    print(f"\n⏱️  Total duration: {duration/60:.1f} minutes")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check overall success
    total_models = len(tickers) * 2  # 2 model types
    successful_models = sum(len([t for t, r in results[mt].items() if r is not None]) 
                           for mt in ['cnn_bilstm', 'transformer'])
    
    success_rate = successful_models / total_models * 100
    
    if success_rate >= 80:
        print(f"\n🎉 RETRAINING SUCCESSFUL! ({success_rate:.1f}% success rate)")
        print("💡 Next steps:")
        print("   1. Run: python check_results.py")
        print("   2. Test web app: streamlit run app.py")
        print("   3. Check logs for constant prediction warnings")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS ({success_rate:.1f}% success rate)")
        print("💡 Recommended actions:")
        print("   1. Check failed tickers individually")
        print("   2. Verify data quality")
        print("   3. Consider adjusting hyperparameters")
    
    return results

def main():
    """Main function"""
    try:
        results = retrain_all_models()
        return results
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return None

if __name__ == "__main__":
    main()
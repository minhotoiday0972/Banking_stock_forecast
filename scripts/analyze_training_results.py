#!/usr/bin/env python3
"""
Script to analyze training results and model performance
"""
import sys
import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("analyzer")

class TrainingAnalyzer:
    """Analyze training results from logs"""
    
    def __init__(self):
        self.config = get_config()
        self.results = defaultdict(lambda: defaultdict(dict))
        self.tickers = self.config.tickers
        self.models = ['cnn_bilstm', 'transformer', 'lstm']
    
    def parse_log_file(self, log_path):
        """Parse training log file to extract metrics"""
        if not os.path.exists(log_path):
            logger.warning(f"Log file not found: {log_path}")
            return
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match training completion and final metrics
        patterns = {
            'final_metrics': r'(\w+) - (\w+) Final: Train Loss: ([\d.]+), Val Loss: ([\d.]+), Test Loss: ([\d.]+), Test RMSE: ([\d.]+), Test MAE: ([\d.]+), Test R²: ([-\d.]+)',
            'epoch_metrics': r'(\w+) - (\w+) Epoch (\d+)/\d+, Train Loss: ([\d.]+), Val Loss: ([\d.]+)',
            'target_metrics': r'Target_(\w+)_t\+(\d+) - (\w+): ([\d.]+)',
            'early_stopping': r'(\w+) - (\w+) Early stopping at epoch (\d+)'
        }
        
        # Extract final metrics
        final_matches = re.findall(patterns['final_metrics'], content)
        for match in final_matches:
            ticker, model, train_loss, val_loss, test_loss, test_rmse, test_mae, test_r2 = match
            self.results[ticker][model]['final'] = {
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'test_loss': float(test_loss),
                'test_rmse': float(test_rmse),
                'test_mae': float(test_mae),
                'test_r2': float(test_r2)
            }
        
        # Extract early stopping info
        early_stopping_matches = re.findall(patterns['early_stopping'], content)
        for match in early_stopping_matches:
            ticker, model, epoch = match
            if ticker in self.results and model in self.results[ticker]:
                self.results[ticker][model]['early_stopping_epoch'] = int(epoch)
    
    def parse_detailed_metrics(self, log_path):
        """Parse detailed metrics from models log"""
        if not os.path.exists(log_path):
            logger.warning(f"Detailed log file not found: {log_path}")
            return
        
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_ticker = None
        current_model = None
        
        for line in lines:
            # Detect model training start
            if "Using device:" in line:
                continue
            
            # Extract epoch metrics
            epoch_match = re.search(r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Val Loss: ([\d.]+)', line)
            if epoch_match:
                epoch, train_loss, val_loss = epoch_match.groups()
                # This indicates start of new model training, but we need to identify which
                continue
            
            # Extract target-specific metrics
            target_match = re.search(r'Target_(\w+)_t\+(\d+) - (\w+): ([\d.-]+)', line)
            if target_match:
                target_type, horizon, metric_name, value = target_match.groups()
                # Store detailed metrics if needed
                continue
            
            # Extract early stopping
            if "Early stopping at epoch" in line:
                epoch_match = re.search(r'Early stopping at epoch (\d+)', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    # Store early stopping info
                    continue
    
    def analyze_results(self):
        """Analyze and summarize training results"""
        # Parse main log files
        logs_dir = "logs"
        self.parse_log_file(os.path.join(logs_dir, "models_20250929.log"))
        
        # Create summary statistics
        summary = {
            'total_models': 0,
            'successful_models': 0,
            'avg_metrics': defaultdict(list),
            'best_models': {},
            'model_comparison': defaultdict(lambda: defaultdict(list))
        }
        
        # Analyze each ticker and model
        for ticker in self.tickers:
            for model in self.models:
                if ticker in self.results and model in self.results[ticker]:
                    if 'final' in self.results[ticker][model]:
                        summary['successful_models'] += 1
                        metrics = self.results[ticker][model]['final']
                        
                        # Collect metrics for averaging
                        for metric, value in metrics.items():
                            summary['avg_metrics'][metric].append(value)
                            summary['model_comparison'][model][metric].append(value)
                
                summary['total_models'] += 1
        
        # Calculate averages
        for metric, values in summary['avg_metrics'].items():
            if values:
                summary['avg_metrics'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Calculate model averages
        for model in summary['model_comparison']:
            for metric in summary['model_comparison'][model]:
                values = summary['model_comparison'][model][metric]
                if values:
                    summary['model_comparison'][model][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
        
        return summary
    
    def print_analysis(self):
        """Print comprehensive analysis"""
        print("=" * 80)
        print("🔍 TRAINING RESULTS ANALYSIS")
        print("=" * 80)
        
        # Check if we have results
        if not self.results:
            print("❌ No training results found in logs")
            print("\nPossible reasons:")
            print("- Training not completed yet")
            print("- Log files not found")
            print("- Different log format")
            return
        
        summary = self.analyze_results()
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS")
        print(f"Total models trained: {summary['total_models']}")
        print(f"Successful models: {summary['successful_models']}")
        print(f"Success rate: {summary['successful_models']/summary['total_models']*100:.1f}%")
        
        # Model comparison
        print(f"\n🤖 MODEL COMPARISON")
        print("-" * 60)
        
        model_names = {'cnn_bilstm': 'CNN-BiLSTM', 'transformer': 'Transformer', 'lstm': 'LSTM'}
        
        for model, model_data in summary['model_comparison'].items():
            if model_data:
                print(f"\n{model_names.get(model, model).upper()}:")
                if 'test_r2' in model_data:
                    r2_stats = model_data['test_r2']
                    print(f"  R² Score: {r2_stats['mean']:.4f} ± {r2_stats['std']:.4f} ({r2_stats['count']} models)")
                
                if 'test_rmse' in model_data:
                    rmse_stats = model_data['test_rmse']
                    print(f"  RMSE: {rmse_stats['mean']:.4f} ± {rmse_stats['std']:.4f}")
                
                if 'test_mae' in model_data:
                    mae_stats = model_data['test_mae']
                    print(f"  MAE: {mae_stats['mean']:.4f} ± {mae_stats['std']:.4f}")
        
        # Detailed results by ticker
        print(f"\n📈 DETAILED RESULTS BY TICKER")
        print("-" * 80)
        
        for ticker in sorted(self.tickers):
            if ticker in self.results:
                print(f"\n🏦 {ticker}:")
                
                for model in self.models:
                    if model in self.results[ticker] and 'final' in self.results[ticker][model]:
                        metrics = self.results[ticker][model]['final']
                        early_stop = self.results[ticker][model].get('early_stopping_epoch', 'N/A')
                        
                        print(f"  {model_names.get(model, model)}:")
                        print(f"    R² Score: {metrics['test_r2']:.4f}")
                        print(f"    RMSE: {metrics['test_rmse']:.4f}")
                        print(f"    MAE: {metrics['test_mae']:.4f}")
                        print(f"    Early Stop: Epoch {early_stop}")
                    else:
                        print(f"  {model_names.get(model, model)}: ❌ No results")
        
        # Performance insights
        print(f"\n💡 PERFORMANCE INSIGHTS")
        print("-" * 60)
        
        # Find best performing models
        best_r2_models = []
        best_rmse_models = []
        
        for ticker in self.results:
            for model in self.results[ticker]:
                if 'final' in self.results[ticker][model]:
                    metrics = self.results[ticker][model]['final']
                    best_r2_models.append((ticker, model, metrics['test_r2']))
                    best_rmse_models.append((ticker, model, metrics['test_rmse']))
        
        if best_r2_models:
            best_r2_models.sort(key=lambda x: x[2], reverse=True)
            print(f"\n🏆 TOP 5 MODELS BY R² SCORE:")
            for i, (ticker, model, r2) in enumerate(best_r2_models[:5]):
                print(f"  {i+1}. {ticker} - {model_names.get(model, model)}: {r2:.4f}")
        
        if best_rmse_models:
            best_rmse_models.sort(key=lambda x: x[2])
            print(f"\n🎯 TOP 5 MODELS BY RMSE (Lower is better):")
            for i, (ticker, model, rmse) in enumerate(best_rmse_models[:5]):
                print(f"  {i+1}. {ticker} - {model_names.get(model, model)}: {rmse:.4f}")
        
        # Model architecture insights
        print(f"\n🔧 MODEL ARCHITECTURE INSIGHTS:")
        
        architecture_performance = {}
        for model in self.models:
            if model in summary['model_comparison'] and 'test_r2' in summary['model_comparison'][model]:
                r2_mean = summary['model_comparison'][model]['test_r2']['mean']
                architecture_performance[model] = r2_mean
        
        if architecture_performance:
            sorted_archs = sorted(architecture_performance.items(), key=lambda x: x[1], reverse=True)
            print(f"  Best performing architecture: {model_names.get(sorted_archs[0][0], sorted_archs[0][0])} (R² = {sorted_archs[0][1]:.4f})")
            
            for model, r2 in sorted_archs:
                print(f"  - {model_names.get(model, model)}: {r2:.4f}")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        
        if summary['successful_models'] < summary['total_models']:
            print(f"  ⚠️ Some models failed to train - check individual logs")
        
        # Check for negative R² scores
        negative_r2_count = 0
        for ticker in self.results:
            for model in self.results[ticker]:
                if 'final' in self.results[ticker][model]:
                    if self.results[ticker][model]['final']['test_r2'] < 0:
                        negative_r2_count += 1
        
        if negative_r2_count > 0:
            print(f"  ⚠️ {negative_r2_count} models have negative R² scores - consider:")
            print(f"    - Increasing training epochs")
            print(f"    - Adjusting learning rate")
            print(f"    - Feature engineering improvements")
            print(f"    - Data preprocessing review")
        
        # Check for overfitting
        high_variance_models = []
        for ticker in self.results:
            for model in self.results[ticker]:
                if 'final' in self.results[ticker][model]:
                    metrics = self.results[ticker][model]['final']
                    if metrics['train_loss'] < metrics['val_loss'] * 0.5:  # Significant gap
                        high_variance_models.append(f"{ticker}-{model}")
        
        if high_variance_models:
            print(f"  ⚠️ Potential overfitting detected in {len(high_variance_models)} models")
            print(f"    - Consider increasing regularization")
            print(f"    - Reduce model complexity")
            print(f"    - Increase training data")

def main():
    """Main function"""
    analyzer = TrainingAnalyzer()
    analyzer.print_analysis()

if __name__ == "__main__":
    main()
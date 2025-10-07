# src/training/trainer.py
import argparse
import numpy as np
import os
from typing import List, Dict, Any, Optional

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..models.cnn_bilstm import train_cnn_bilstm_model
from ..models.transformer import train_transformer_model

logger = get_logger("trainer")

class ModelTrainingPipeline:
    """Centralized model training pipeline"""
    
    def __init__(self):
        self.config = get_config()
        self.available_models = {
            'cnn_bilstm': train_cnn_bilstm_model,
            'transformer': train_transformer_model
        }
    
    def train_single_model(self, model_type: str, ticker: str) -> Optional[Dict[str, Any]]:
        """Train a single model for a ticker"""
        if model_type not in self.available_models:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        try:
            logger.info(f"Training {model_type} for {ticker}")
            train_func = self.available_models[model_type]
            result = train_func(ticker)
            logger.info(f"Successfully trained {model_type} for {ticker}")
            return result
        except Exception as e:
            logger.error(f"Failed to train {model_type} for {ticker}: {e}")
            return None
    
    def train_all_models(self, model_types: List[str] = None, 
                        tickers: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Train all specified models for all tickers"""
        if model_types is None:
            model_types = list(self.available_models.keys())
        if tickers is None:
            tickers = self.config.tickers
        
        logger.info(f"Starting training pipeline for {len(model_types)} models and {len(tickers)} tickers")
        
        results = {}
        
        for model_type in model_types:
            if model_type not in self.available_models:
                logger.warning(f"Skipping unknown model type: {model_type}")
                continue
            
            results[model_type] = {}
            
            for ticker in tickers:
                # Check if sequences file exists
                sequences_path = os.path.join(self.config.processed_dir, f"{ticker}_sequences.npz")
                if not os.path.exists(sequences_path):
                    logger.warning(f"No sequences file for {ticker}, skipping")
                    results[model_type][ticker] = None
                    continue
                
                result = self.train_single_model(model_type, ticker)
                results[model_type][ticker] = result
        
        self._print_summary(results)
        return results
    
    def _print_summary(self, results: Dict[str, Dict[str, Any]]):
        """Print training summary"""
        print(f"\n{'='*50}")
        print("TRAINING SUMMARY")
        print(f"{'='*50}")
        
        for model_type, model_results in results.items():
            successful = [t for t, r in model_results.items() if r is not None]
            failed = [t for t, r in model_results.items() if r is None]
            
            print(f"\n{model_type.upper()}:")
            print(f"  Successful: {len(successful)}/{len(model_results)} ({len(successful)/len(model_results)*100:.1f}%)")
            print(f"  Successful tickers: {successful}")
            if failed:
                print(f"  Failed tickers: {failed}")
            
            # Print best metrics for successful models
            if successful:
                print(f"  Best test metrics:")
                for ticker in successful[:3]:  # Show first 3
                    result = model_results[ticker]
                    if result and 'test_metrics' in result:
                        test_metrics = result['test_metrics']
                        for target_name, metrics in test_metrics.items():
                            if metrics['type'] == 'regression':
                                print(f"    {ticker} {target_name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
                            else:
                                print(f"    {ticker} {target_name}: Accuracy={metrics['accuracy']:.4f}")

def main():
    """Main function for model training"""
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument('--models', nargs='+', default=['cnn_bilstm'], 
                       choices=['cnn_bilstm', 'transformer', 'lstm', 'all'],
                       help="Models to train")
    parser.add_argument('--tickers', nargs='+', default=None,
                       help="Tickers to train (default: all from config)")
    parser.add_argument('--config', default='config.yaml',
                       help="Config file path")
    
    args = parser.parse_args()
    
    # Handle 'all' option
    if 'all' in args.models:
        args.models = ['cnn_bilstm', 'transformer', 'lstm']
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline()
    
    # Run training
    results = pipeline.train_all_models(args.models, args.tickers)
    
    logger.info("Training pipeline completed")

if __name__ == "__main__":
    main()
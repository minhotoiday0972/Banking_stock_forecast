import torch
import torch.nn as nn
from typing import Dict, Any

from .base_model import BaseModel

class CNNBiLSTM(BaseModel):
    """CNN-BiLSTM model for multi-horizon forecasting"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super(CNNBiLSTM, self).__init__(input_dim, config)
        
        # Model parameters
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.kernel_size = config.get('kernel_size', 3)
        self.dropout_rate = config.get('dropout_rate', 0.4)
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_dim, self.hidden_dim, 
                              kernel_size=self.kernel_size, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        # BiLSTM layer
        self.bilstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.num_layers,
            bidirectional=True, batch_first=True, dropout=self.dropout_rate
        )
        
        # Output heads for different horizons and tasks
        self.regression_heads = nn.ModuleDict()
        self.classification_heads = nn.ModuleDict()
        
        for horizon in self.horizons:
            # Regression head (price prediction)
            self.regression_heads[f'Target_Close_t+{horizon}'] = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, 1)
            )
            
            # Classification head (direction prediction)
            self.classification_heads[f'Target_Direction_t+{horizon}'] = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, 3)  # 3 classes: Down, Flat, Up
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # CNN feature extraction
        x = x.transpose(1, 2)  # (batch, features, time)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        # Use last time step output
        features = lstm_out[:, -1, :]  # (batch, hidden_dim * 2)
        
        # Generate outputs for all horizons
        outputs = {}
        
        # Regression outputs
        for key, head in self.regression_heads.items():
            outputs[key] = head(features)
        
        # Classification outputs
        for key, head in self.classification_heads.items():
            outputs[key] = head(features)
        
        return outputs

def train_cnn_bilstm_model(ticker: str) -> Dict[str, Any]:
    """Train CNN-BiLSTM model for a ticker"""
    from .base_model import ModelTrainer
    from ..utils.config import get_config
    import numpy as np
    import os
    
    config = get_config()
    trainer = ModelTrainer("cnn_bilstm")
    
    # Load sequences
    sequences_path = os.path.join(config.processed_dir, f"{ticker}_sequences.npz")
    sequences = dict(np.load(sequences_path))
    
    # Prepare data
    data = trainer.prepare_data(
        sequences,
        train_split=config.get('training.train_split', 0.8),
        val_split=config.get('training.val_split', 0.1)
    )
    
    # Create model
    model_config = config.get('models.cnn_bilstm', {})
    model_config['forecast_horizons'] = config.get('models.forecast_horizons', [1, 3, 5])
    model = CNNBiLSTM(data['input_dim'], model_config)
    
    # Train model
    results = trainer.train_model(
        model, data,
        epochs=config.get('training.epochs', 50),
        batch_size=config.get('training.batch_size', 32),
        learning_rate=0.001,
        early_stopping_patience=config.get('training.early_stopping_patience', 10)
    )
    
    # Save model and plots
    trainer.save_model(results['model'], ticker, "cnn_bilstm")
    trainer.plot_training_history(results['history'], ticker, "CNN-BiLSTM")
    
    return results

def main():
    """Main function for CNN-BiLSTM training"""
    from ..utils.config import get_config
    from ..utils.logger import get_logger
    
    config = get_config()
    logger = get_logger("cnn_bilstm")
    
    tickers = config.tickers
    results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Training CNN-BiLSTM for {ticker}")
            result = train_cnn_bilstm_model(ticker)
            results[ticker] = result
            logger.info(f"Successfully trained CNN-BiLSTM for {ticker}")
        except Exception as e:
            logger.error(f"Failed to train CNN-BiLSTM for {ticker}: {e}")
            results[ticker] = None
    
    # Summary
    successful = [t for t, r in results.items() if r is not None]
    failed = [t for t, r in results.items() if r is None]
    
    print(f"\n=== CNN-BiLSTM Training Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {len(successful)}/{len(tickers)} ({len(successful)/len(tickers)*100:.1f}%)")

if __name__ == "__main__":
    main()
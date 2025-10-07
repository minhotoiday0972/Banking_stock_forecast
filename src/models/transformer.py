import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from .base_model import BaseModel

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(BaseModel):
    """Transformer model for multi-horizon forecasting"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super(TransformerModel, self).__init__(input_dim, config)
        
        # Model parameters
        self.d_model = config.get('d_model', 64)
        self.nhead = config.get('nhead', 4)
        self.num_layers = config.get('num_layers', 2)
        self.dim_feedforward = config.get('dim_feedforward', 128)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Input projection
        self.input_fc = nn.Linear(input_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, self.num_layers)
        
        # Output heads for different horizons and tasks
        self.regression_heads = nn.ModuleDict()
        self.classification_heads = nn.ModuleDict()
        
        for horizon in self.horizons:
            # Regression head (price prediction)
            self.regression_heads[f'Target_Close_t+{horizon}'] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.d_model // 2, 1)
            )
            
            # Classification head (direction prediction)
            self.classification_heads[f'Target_Direction_t+{horizon}'] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.d_model // 2, 3)  # 3 classes: Down, Flat, Up
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Input projection
        x = self.input_fc(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Use last time step output
        features = x[:, -1, :]  # (batch, d_model)
        
        # Generate outputs for all horizons
        outputs = {}
        
        # Regression outputs
        for key, head in self.regression_heads.items():
            outputs[key] = head(features)
        
        # Classification outputs
        for key, head in self.classification_heads.items():
            outputs[key] = head(features)
        
        return outputs

def train_transformer_model(ticker: str) -> Dict[str, Any]:
    """Train Transformer model for a ticker"""
    from .base_model import ModelTrainer
    from ..utils.config import get_config
    import numpy as np
    import os
    
    config = get_config()
    trainer = ModelTrainer("transformer")
    
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
    model_config = config.get('models.transformer', {})
    model_config['forecast_horizons'] = config.get('models.forecast_horizons', [1, 3, 5])
    model = TransformerModel(data['input_dim'], model_config)
    
    # Train model
    results = trainer.train_model(
        model, data,
        epochs=config.get('training.epochs', 50),
        batch_size=config.get('training.batch_size', 32),
        learning_rate=0.001,
        early_stopping_patience=config.get('training.early_stopping_patience', 10)
    )
    
    # Save model and plots
    trainer.save_model(results['model'], ticker, "transformer")
    trainer.plot_training_history(results['history'], ticker, "Transformer")
    
    return results

def main():
    """Main function for Transformer training"""
    from ..utils.config import get_config
    from ..utils.logger import get_logger
    
    config = get_config()
    logger = get_logger("transformer")
    
    tickers = config.tickers
    results = {}
    
    for ticker in tickers:
        try:
            logger.info(f"Training Transformer for {ticker}")
            result = train_transformer_model(ticker)
            results[ticker] = result
            logger.info(f"Successfully trained Transformer for {ticker}")
        except Exception as e:
            logger.error(f"Failed to train Transformer for {ticker}: {e}")
            results[ticker] = None
    
    # Summary
    successful = [t for t, r in results.items() if r is not None]
    failed = [t for t, r in results.items() if r is None]
    
    print(f"\n=== Transformer Training Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {len(successful)}/{len(tickers)} ({len(successful)/len(tickers)*100:.1f}%)")

if __name__ == "__main__":
    main()
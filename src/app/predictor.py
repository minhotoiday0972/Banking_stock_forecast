# src/app/predictor.py
import torch
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Tuple
import os

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_database
from ..models.cnn_bilstm import CNNBiLSTM
from ..models.transformer import TransformerModel

logger = get_logger("predictor")

class StockPredictor:
    """Stock prediction interface"""
    
    def __init__(self):
        self.config = get_config()
        self.db = get_database()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.scalers = {}
        self.feature_columns = self._get_feature_columns()
    
    def _get_feature_columns(self):
        """Get feature columns from config"""
        technical = self.config.get('features.technical', [])
        fundamental = self.config.get('features.fundamental', [])
        banking_specific = self.config.get('features.banking_specific', [])
        market = self.config.get('features.market', [])
        derived = ['MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Momentum_20D']
        return technical + fundamental + banking_specific + market + derived
    
    def load_model(self, ticker: str, model_type: str) -> bool:
        """Load trained model for ticker"""
        try:
            model_path = os.path.join(self.config.models_dir, f"{ticker}_{model_type}_best.pt")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model config
            if model_type == 'cnn_bilstm':
                model_config = self.config.get('models.cnn_bilstm', {})
                model_config['forecast_horizons'] = self.config.get('models.forecast_horizons', [1, 3, 5])
                model = CNNBiLSTM(len(self.feature_columns), model_config)
            elif model_type == 'transformer':
                model_config = self.config.get('models.transformer', {})
                model_config['forecast_horizons'] = self.config.get('models.forecast_horizons', [1, 3, 5])
                model = TransformerModel(len(self.feature_columns), model_config)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            # Load state dict with weights_only=True for security
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            model.to(self.device)
            model.eval()
            
            self.models[f"{ticker}_{model_type}"] = model
            logger.info(f"Loaded model: {ticker}_{model_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {ticker}_{model_type}: {e}")
            return False
    
    def load_scalers(self, ticker: str) -> bool:
        """Load scalers for ticker"""
        try:
            scalers_path = os.path.join(self.config.processed_dir, f"{ticker}_scalers.pkl")
            if not os.path.exists(scalers_path):
                logger.error(f"Scalers file not found: {scalers_path}")
                return False
            
            scalers = joblib.load(scalers_path)
            self.scalers[ticker] = scalers
            logger.info(f"Loaded scalers for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load scalers for {ticker}: {e}")
            return False
    
    def get_latest_features(self, ticker: str, timesteps: int = None) -> Optional[np.ndarray]:
        """Get latest features for prediction"""
        if timesteps is None:
            timesteps = self.config.get('training.timesteps', 30)
        
        try:
            # Load processed features
            features_path = os.path.join(self.config.processed_dir, f"{ticker}_features.csv")
            if not os.path.exists(features_path):
                logger.error(f"Features file not found: {features_path}")
                return None
            
            df = pd.read_csv(features_path)
            
            # Get available feature columns
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            if len(df) < timesteps:
                logger.error(f"Not enough data for {ticker}: {len(df)} < {timesteps}")
                return None
            
            # Get last timesteps rows
            features = df[available_features].tail(timesteps).values
            
            # Reshape for model input: (1, timesteps, features)
            features = features.reshape(1, timesteps, -1)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to get features for {ticker}: {e}")
            return None
    
    def predict(self, ticker: str, model_type: str, horizon: int = 1) -> Optional[Dict[str, Any]]:
        """Make prediction for ticker"""
        model_key = f"{ticker}_{model_type}"
        
        # Check if model is loaded
        if model_key not in self.models:
            if not self.load_model(ticker, model_type):
                return None
        
        # Check if scalers are loaded
        if ticker not in self.scalers:
            if not self.load_scalers(ticker):
                return None
        
        # Get features
        features = self.get_latest_features(ticker)
        if features is None:
            return None
        
        try:
            model = self.models[model_key]
            
            # Convert to tensor
            X = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(X)
            
            # Process outputs
            results = {}
            
            # Regression prediction (price)
            price_key = f'Target_Close_t+{horizon}'
            if price_key in outputs:
                price_pred = outputs[price_key].cpu().numpy()[0, 0]
                
                # Inverse transform if scaler available
                if price_key in self.scalers[ticker]:
                    scaler = self.scalers[ticker][price_key]
                    price_pred = scaler.inverse_transform([[price_pred]])[0, 0]
                
                results['predicted_price'] = float(price_pred)
            
            # Classification prediction (direction)
            direction_key = f'Target_Direction_t+{horizon}'
            if direction_key in outputs:
                direction_logits = outputs[direction_key].cpu().numpy()[0]
                direction_probs = torch.softmax(torch.tensor(direction_logits), dim=0).numpy()
                
                # Post-process based on actual price change (will be calculated later)
                # Store original probabilities for now
                original_probs = direction_probs.copy()
                
                direction_labels = ['Down', 'Flat', 'Up']
                predicted_direction = direction_labels[np.argmax(direction_probs)]
                confidence = float(np.max(direction_probs))
                
                results['predicted_direction'] = predicted_direction
                results['direction_confidence'] = confidence
                results['direction_probabilities'] = {
                    label: float(prob) for label, prob in zip(direction_labels, direction_probs)
                }
            
            # Add price change information
            if 'predicted_price' in results:
                # Get actual current price from historical data (not scaled)
                historical_data = self.get_historical_data(ticker, days=1)
                if historical_data is not None and not historical_data.empty:
                    current_price = float(historical_data['Close'].iloc[-1])
                else:
                    current_price = float(X[0, -1, 3])  # Fallback to scaled data
                
                predicted_price = results['predicted_price']
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                results['current_price'] = current_price
                results['price_change'] = price_change
                results['price_change_pct'] = price_change_pct
            
            # Post-process direction probabilities based on actual price change
            if 'predicted_price' in results and 'current_price' in results and direction_key in outputs:
                current_price = results['current_price']
                predicted_price = results['predicted_price']
                price_change_pct = (predicted_price - current_price) / current_price
                
                logger.info(f"Post-processing: Current={current_price:.2f}, Predicted={predicted_price:.2f}, Change={price_change_pct:.4f}")
                
                # Get direction probabilities
                direction_probs = np.array([
                    results['direction_probabilities']['Down'],
                    results['direction_probabilities']['Flat'], 
                    results['direction_probabilities']['Up']
                ])
                
                logger.info(f"Original probs: Down={direction_probs[0]:.3f}, Flat={direction_probs[1]:.3f}, Up={direction_probs[2]:.3f}")
                
                # Strong adjustment for large price changes
                if abs(price_change_pct) > 0.10:  # If change > 10% - extremely strong
                    if price_change_pct < -0.10:  # Extremely strong down
                        direction_probs = np.array([0.85, 0.10, 0.05])  # Force Down
                        logger.info("Applied extreme down adjustment (>10%)")
                    elif price_change_pct > 0.10:  # Extremely strong up
                        direction_probs = np.array([0.05, 0.10, 0.85])  # Force Up
                        logger.info("Applied extreme up adjustment (>10%)")
                elif abs(price_change_pct) > 0.05:  # If change > 5% - very strong
                    if price_change_pct < -0.05:  # Very strong down
                        direction_probs = np.array([0.70, 0.20, 0.10])  # Force Down
                        logger.info("Applied very strong down adjustment (>5%)")
                    elif price_change_pct > 0.05:  # Very strong up
                        direction_probs = np.array([0.10, 0.20, 0.70])  # Force Up
                        logger.info("Applied very strong up adjustment (>5%)")
                elif abs(price_change_pct) > 0.02:  # If change > 2% - strong
                    if price_change_pct < -0.02:  # Strong down
                        direction_probs[0] = min(0.80, direction_probs[0] + 0.6)  # Down
                        direction_probs[1] = max(0.10, direction_probs[1] - 0.5)  # Flat
                        direction_probs[2] = max(0.10, direction_probs[2] - 0.1)  # Up
                        direction_probs = direction_probs / direction_probs.sum()
                        logger.info("Applied strong down adjustment (>2%)")
                    elif price_change_pct > 0.02:  # Strong up
                        direction_probs[2] = min(0.80, direction_probs[2] + 0.6)  # Up
                        direction_probs[1] = max(0.10, direction_probs[1] - 0.5)  # Flat
                        direction_probs[0] = max(0.10, direction_probs[0] - 0.1)  # Down
                        direction_probs = direction_probs / direction_probs.sum()
                        logger.info("Applied strong up adjustment (>2%)")
                
                logger.info(f"Final probs: Down={direction_probs[0]:.3f}, Flat={direction_probs[1]:.3f}, Up={direction_probs[2]:.3f}")
                
                # Update results
                direction_labels = ['Down', 'Flat', 'Up']
                results['predicted_direction'] = direction_labels[np.argmax(direction_probs)]
                results['direction_confidence'] = float(np.max(direction_probs))
                results['direction_probabilities'] = {
                    label: float(prob) for label, prob in zip(direction_labels, direction_probs)
                }
            
            results['horizon'] = horizon
            results['model_type'] = model_type
            results['ticker'] = ticker
            
            logger.info(f"Prediction made for {ticker} using {model_type}, horizon {horizon}")
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}_{model_type}: {e}")
            return None
    
    def get_available_models(self, ticker: str) -> list:
        """Get list of available models for ticker"""
        available = []
        model_types = ['cnn_bilstm', 'transformer']
        
        for model_type in model_types:
            model_path = os.path.join(self.config.models_dir, f"{ticker}_{model_type}_best.pt")
            if os.path.exists(model_path):
                available.append(model_type)
        
        return available
    
    def get_historical_data(self, ticker: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical price data for visualization"""
        try:
            query = f"""
            SELECT time, Open, High, Low, Close, Volume 
            FROM {ticker}_OHLCV 
            ORDER BY time DESC 
            LIMIT {days}
            """
            df = self.db.load_dataframe(f"{ticker}_OHLCV", query)
            
            if df is not None:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time')
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {ticker}: {e}")
            return None
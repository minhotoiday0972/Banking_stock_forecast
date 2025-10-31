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
    def __init__(self):
        self.config = get_config()
        self.db = get_database()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.metadata_cache = {}

    def _load_metadata(self, ticker: str) -> bool:
        """Tải metadata (scalers và feature_columns) từ file."""
        if ticker in self.metadata_cache:
            return True
        try:
            metadata_path = os.path.join(self.config.get('paths.processed'), f"{ticker}_metadata.pkl")
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found: {metadata_path}. Please run 'python main.py features'.")
                return False
            
            self.metadata_cache[ticker] = joblib.load(metadata_path)
            logger.info(f"Loaded metadata for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Failed to load metadata for {ticker}: {e}")
            return False

    def load_model(self, ticker: str, model_type: str) -> bool:
        """Tải model đã huấn luyện, sử dụng metadata để đảm bảo tính nhất quán."""
        model_key = f"{ticker}_{model_type}"
        if model_key in self.models:
            return True
        try:
            if not self._load_metadata(ticker):
                return False
            
            metadata = self.metadata_cache[ticker]
            feature_columns = metadata.get('feature_columns')
            if not feature_columns:
                logger.error(f"Feature columns not found in metadata for {ticker}.")
                return False
            
            input_dim = len(feature_columns)
            model_path = os.path.join(self.config.get('paths.models'), f"{ticker}_{model_type}_best.pt")
            
            model_config = self.config.get('models', {}).get(model_type, {})
            shared_config = self.config.get('models', {}).get('shared', {})
            full_config = {**shared_config, **model_config}
            
            model_class = {'cnn_bilstm': CNNBiLSTM, 'transformer': TransformerModel}.get(model_type)
            if not model_class:
                 raise ValueError(f"Unknown model type: {model_type}")

            model = model_class(input_dim, self.config.get('models'))

            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models[model_key] = model
            logger.info(f"Successfully loaded model {model_key} with {input_dim} features.")
            return True
        except Exception as e:
            logger.error(f"Failed to load model {model_key}: {e}", exc_info=True)
            return False

    def get_latest_features(self, ticker: str) -> Optional[np.ndarray]:
        """Lấy các đặc trưng gần nhất để dự báo."""
        try:
            timesteps = self.config.get('training.timesteps', 30)
            metadata = self.metadata_cache.get(ticker)
            if not metadata:
                logger.error(f"Metadata not loaded for {ticker}.")
                return None
            
            feature_cols = metadata['feature_columns']
            features_path = os.path.join(self.config.get('paths.processed'), f"{ticker}_features_scaled.csv")
            df = pd.read_csv(features_path)
            
            if len(df) < timesteps:
                logger.error(f"Not enough data for {ticker}: {len(df)} < {timesteps}")
                return None
            
            features = df[feature_cols].tail(timesteps).values
            return features.reshape(1, timesteps, -1)
        except Exception as e:
            logger.error(f"Failed to get features for {ticker}: {e}")
            return None

    def predict(self, ticker: str, model_type: str, horizon: int = 1) -> Optional[Dict[str, Any]]:
        """Thực hiện dự báo cho một mã cổ phiếu."""
        model_key = f"{ticker}_{model_type}"
        
        if not self.load_model(ticker, model_type):
            return None
        
        features = self.get_latest_features(ticker)
        if features is None:
            return None
        
        try:
            model = self.models[model_key]
            X = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                outputs = model(X)
            
            results = {}
            metadata = self.metadata_cache[ticker]
            
            price_key = f'Target_Close_t+{horizon}'
            if price_key in outputs and price_key in metadata['scalers']:
                price_pred = outputs[price_key].cpu().numpy()[0, 0]
                scaler = metadata['scalers'][price_key]
                results['predicted_price'] = float(scaler.inverse_transform([[price_pred]])[0, 0])
            
            direction_key = f'Target_Direction_t+{horizon}'
            if direction_key in outputs:
                direction_logits = outputs[direction_key].cpu().numpy()[0]
                direction_probs = torch.softmax(torch.tensor(direction_logits), dim=0).numpy()
                direction_labels = ['Down', 'Up'] 
                results['predicted_direction'] = direction_labels[np.argmax(direction_probs)]
                results['direction_confidence'] = float(np.max(direction_probs))
                results['direction_probabilities'] = {
                    label: float(prob) for label, prob in zip(direction_labels, direction_probs)
                }
            
            # Lấy giá hiện tại
            historical_data = self.get_historical_data(ticker, days=1)
            if historical_data is not None and not historical_data.empty:
                current_price = float(historical_data['Close'].iloc[-1])
                results['current_price'] = current_price
                if 'predicted_price' in results:
                    results['price_change_pct'] = ((results['predicted_price'] - current_price) / current_price)

            return results
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_key}: {e}", exc_info=True)
            return None
    
    def get_available_models(self, ticker: str) -> list:
        """Lấy danh sách các model có sẵn cho một mã cổ phiếu."""
        available = []
        model_types = list(self.config.get('models', {}).keys())
        model_types = [m for m in model_types if m != 'shared']
        
        for model_type in model_types:
            model_path = os.path.join(self.config.get('paths.models'), f"{ticker}_{model_type}_best.pt")
            if os.path.exists(model_path):
                available.append(model_type)
        return available
    
    def get_historical_data(self, ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
        """Lấy dữ liệu giá lịch sử để vẽ biểu đồ."""
        try:
            df = self.db.load_dataframe(f"{ticker}_OHLCV")
            if df is not None:
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time').tail(days)
                return df
            return None
        except Exception as e:
            logger.error(f"Failed to get historical data for {ticker}: {e}")
            return None
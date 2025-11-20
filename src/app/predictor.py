# src/app/predictor.py
import torch
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional
import os

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_database
# --- THAY ĐỔI: Import các model đã được đơn giản hóa ---
from ..models.cnn_bilstm import CNNBiLSTM
from ..models.transformer import TransformerModel

logger = get_logger("predictor")

class StockPredictor:
    def __init__(self):
        self.config = get_config()
        self.db = get_database()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cache cho model và metadata
        self.models_cache = {}
        self.metadata_cache = {}
        self.main_scaler_cache = {} # Cache cho main scaler

    def _load_main_scaler(self, ticker: str) -> Optional[joblib.load]:
        """Tải main scaler (dùng để scale dữ liệu input)."""
        if ticker in self.main_scaler_cache:
            return self.main_scaler_cache[ticker]
            
        scaler_path = os.path.join(self.config.get('paths.processed'), f"{ticker}_main_scaler.pkl")
        if not os.path.exists(scaler_path):
            logger.error(f"Không tìm thấy main scaler: {scaler_path}")
            return None
        
        scaler = joblib.load(scaler_path)
        self.main_scaler_cache[ticker] = scaler
        return scaler

    def _load_model_and_metadata(self, ticker: str, model_type: str, horizon: int) -> bool:
        """Tải model và metadata chuyên biệt cho một horizon."""
        model_key = f"{ticker}_{model_type}_t+{horizon}"
        
        if model_key in self.models_cache:
            return True
        
        try:
            # 1. Tải Metadata chuyên biệt
            metadata_path = os.path.join(self.config.get('paths.processed'), f"{ticker}_metadata_t+{horizon}.pkl")
            if not os.path.exists(metadata_path):
                logger.error(f"Không tìm thấy metadata: {metadata_path}")
                return False
            metadata = joblib.load(metadata_path)
            self.metadata_cache[model_key] = metadata
            
            # 2. Tải Model
            model_path = os.path.join(self.config.get('paths.models'), f"{model_key}_best.pt")
            if not os.path.exists(model_path):
                logger.error(f"Không tìm thấy model: {model_path}")
                return False

            input_dim = len(metadata['feature_columns'])
            
            # Khởi tạo model (đã được đơn giản hóa)
            if model_type == 'cnn_bilstm':
                model = CNNBiLSTM(input_dim, self.config.get('models'))
            elif model_type == 'transformer':
                model = TransformerModel(input_dim, self.config.get('models'))
            else:
                logger.error(f"Model không xác định: {model_type}")
                return False
            
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            self.models_cache[model_key] = model
            logger.info(f"Đã tải model và metadata cho {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải model {model_key}: {e}", exc_info=True)
            return False

    def get_latest_features(self, ticker: str, metadata: Dict) -> Optional[np.ndarray]:
        """Lấy và chuẩn bị chuỗi đặc trưng cuối cùng."""
        try:
            timesteps = self.config.get('training.timesteps', 30)
            feature_cols = metadata['feature_columns']
            
            # Lấy dữ liệu đã được scale TỔNG THỂ
            features_path = os.path.join(self.config.get('paths.processed'), f"{ticker}_features_scaled.csv")
            if not os.path.exists(features_path):
                 logger.error(f"Không tìm thấy file {features_path}")
                 return None
            
            df = pd.read_csv(features_path)
            
            if len(df) < timesteps:
                logger.error(f"Không đủ dữ liệu cho {ticker}: {len(df)} < {timesteps}")
                return None
            
            # Lấy đúng các cột đặc trưng "vàng" (đã được scale)
            features = df[feature_cols].tail(timesteps).values
            
            # Reshape cho model: (1, timesteps, features)
            return features.reshape(1, timesteps, -1)
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy đặc trưng cho {ticker}: {e}")
            return None
    
    def predict(self, ticker: str, model_type: str, horizon: int) -> Optional[Dict[str, Any]]:
        """Thực hiện dự đoán xu hướng."""
        model_key = f"{ticker}_{model_type}_t+{horizon}"
        
        # 1. Tải model và metadata nếu chưa có
        if model_key not in self.models_cache:
            if not self._load_model_and_metadata(ticker, model_type, horizon):
                return None
        
        model = self.models_cache[model_key]
        metadata = self.metadata_cache[model_key]
        
        # 2. Lấy dữ liệu đặc trưng mới nhất
        features = self.get_latest_features(ticker, metadata)
        if features is None:
            return None
            
        try:
            # 3. Dự đoán
            X = torch.tensor(features, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = model(X) # outputs là logits trực tiếp
            
            direction_logits = outputs.cpu().numpy()[0]
            direction_probs = torch.softmax(torch.tensor(direction_logits), dim=0).numpy()
            
            direction_labels = ['Down', 'Up'] # Giả định 0=Down, 1=Up
            predicted_direction = direction_labels[np.argmax(direction_probs)]
            confidence = float(np.max(direction_probs))
            
            results = {
                'predicted_direction': predicted_direction,
                'direction_confidence': confidence,
                'direction_probabilities': {
                    'Down': float(direction_probs[0]),
                    'Up': float(direction_probs[1])
                },
                'horizon': horizon,
                'model_type': model_type,
                'ticker': ticker
            }
            logger.info(f"Dự đoán thành công cho {model_key}")
            return results
            
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán {model_key}: {e}")
            return None

    def get_latest_fundamentals(self, ticker: str) -> Optional[pd.Series]:
        """Lấy các chỉ số cơ bản mới nhất để hiển thị."""
        try:
            # Lấy bản ghi mới nhất
            query = f"SELECT * FROM {ticker}_Fundamental ORDER BY time DESC LIMIT 1"
            df = self.db.load_dataframe(f"{ticker}_Fundamental", query)
            
            if df is None or df.empty:
                logger.warning(f"Không tìm thấy dữ liệu cơ bản cho {ticker}")
                return None
            
            return df.iloc[0] # Trả về 1 Series
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu cơ bản cho {ticker}: {e}")
            return None
    
    def get_available_models(self, ticker: str) -> list:
        """Kiểm tra các model đã huấn luyện."""
        available = []
        model_types = ['cnn_bilstm', 'transformer']
        horizons = self.config.get('models.shared.forecast_horizons', [1, 3, 5, 30, 60, 90])
        
        # Chỉ kiểm tra các model cho các horizon đã định nghĩa
        for model_type in model_types:
            # Chỉ cần 1 model (vd t+1) tồn tại là đủ
            # (Chúng ta giả định nếu t+1 được huấn luyện, các horizon khác cũng được huấn luyện)
            model_path = os.path.join(self.config.get('paths.models'), f"{ticker}_{model_type}_t+{horizons[0]}_best.pt")
            if os.path.exists(model_path):
                available.append(model_type)
        
        return list(set(available)) # Trả về các model type duy nhất

    def get_historical_data(self, ticker: str, days: int = 90) -> Optional[pd.DataFrame]:
        # (Hàm này giữ nguyên như cũ)
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
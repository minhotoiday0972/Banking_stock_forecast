# src/training/train_quarterly.py
"""
Nhánh 2: Huấn luyện mô hình HÀNG QUÝ (Phiên bản DL Nâng cao)
Sử dụng các mô hình từ Nhánh 1 (CNN-BiLSTM, Transformer) để so sánh.
"""
import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..models.base_model import BaseModel
from ..models.cnn_bilstm import CNNBiLSTM
from ..models.transformer import TransformerModel

# --- THAY ĐỔI: Sử dụng logger riêng cho Nhánh 2 ---
logger = get_logger("train_quarterly") # Đổi tên logger để ghi vào file riêng (ví dụ: logs/train_quarterly.log)

# --- Lớp QuarterlyLSTM (Giữ nguyên) ---
class QuarterlyLSTM(nn.Module):
     # ... (code giữ nguyên) ...
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.3):
        super(QuarterlyLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 2) # 2 Lớp (Down/Up)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        # --- THAY ĐỔI: Trả về dict để tương thích ---
        # Giả định mục tiêu luôn là t+1Q cho nhánh này
        return {'Target_Direction_t+1Q': out}
        # --- KẾT THÚC THAY ĐỔI ---

# --- Lớp QuarterlyModelTrainer (Giữ nguyên) ---
class QuarterlyModelTrainer:
    # ... (code giữ nguyên) ...
    def __init__(self, config):
        self.config = config
        self.q_config = config.get('training_quarterly', {}) # Lấy config của nhánh 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.timesteps = self.q_config.get('timesteps', 4)
        self.epochs = self.q_config.get('epochs', 100)
        self.lr = self.q_config.get('learning_rate', 0.001)
        self.patience = self.q_config.get('early_stopping_patience', 15)
        self.batch_size = self.q_config.get('batch_size', 4)

    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple:
        X, y = [], []
        for i in range(len(df) - self.timesteps + 1): # +1 để bao gồm cả mẫu cuối
            X.append(df[feature_cols].iloc[i:i + self.timesteps].values)
            y.append(df[target_col].iloc[i + self.timesteps - 1])
        return np.array(X), np.array(y).astype(int)

    def _calculate_class_weights(self, target_data: np.ndarray) -> torch.Tensor:
        class_counts = np.bincount(target_data.astype(int), minlength=2)
        total_samples = class_counts.sum()
        if total_samples == 0:
            return torch.ones(2, dtype=torch.float32)
        weights = total_samples / (len(class_counts) * class_counts)
        weights[np.isinf(weights)] = 1.0
        return torch.tensor(weights, dtype=torch.float32).to(self.device)

    def train(self, model: BaseModel, X_train, y_train, X_val, y_val) -> BaseModel:
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        weights = self._calculate_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = model.state_dict() # Lưu trạng thái ban đầu

        model.to(self.device)
        for epoch in range(self.epochs):
            model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)

                loss = criterion(outputs['Target_Direction_t+1Q'], y_batch)

                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t.to(self.device))
                val_loss = criterion(val_outputs['Target_Direction_t+1Q'], y_val_t.to(self.device))

            if (epoch + 1) % 10 == 0:
                 # --- THAY ĐỔI: Sử dụng logger thay vì print ---
                logger.debug(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {total_train_loss/len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}")
                 # --- KẾT THÚC THAY ĐỔI ---

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping tại huấn luyện hàng quý.")
                    break

        model.load_state_dict(best_model_state)
        return model

# --- Lớp QuarterlyTrainingPipeline (Thêm hàm log) ---

class QuarterlyTrainingPipeline:
    def __init__(self):
        self.config = get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_to_train = {
            "cnn_bilstm": CNNBiLSTM,
            "transformer": TransformerModel,
        }
        # --- THAY ĐỔI: Khởi tạo Trainer ở đây ---
        self.trainer = QuarterlyModelTrainer(self.config)
        # --- KẾT THÚC THAY ĐỔI ---

    # --- HÀM MỚI: Log kết quả ---
    def _log_results(self, ticker: str, model_name: str, bal_acc: float):
        """Hàm trợ giúp để in và ghi log kết quả Balanced Accuracy."""
        header = f"--- Kết quả Test {model_name.upper()} cho {ticker} (Hàng quý) ---"
        result_line = f"  Balanced Accuracy: {bal_acc:.2%}"

        print(f"\n{header}")
        print(result_line)
        logger.info(f"\n{header}\n{result_line}") # Ghi vào log file
    # --- KẾT THÚC HÀM MỚI ---

    def train_single_ticker(self, ticker: str, df: pd.DataFrame) -> Dict:
        target_col = 'Target_Direction_t+1Q'
        non_feature_cols = [
            'time', 'Ticker', 'Quarterly_Return', 'Target_Return_t+1Q',
            'Target_Direction_t+1Q'
            # --- THAY ĐỔI: Loại bỏ 'Close' khỏi đây vì nó là đặc trưng ---
        ]
        # --- KẾT THÚC THAY ĐỔI ---

        # Xác định đặc trưng dựa trên các cột còn lại
        feature_cols = [col for col in df.columns if col not in non_feature_cols and pd.api.types.is_numeric_dtype(df[col])]

        if len(feature_cols) == 0:
            logger.warning(f"Bỏ qua {ticker}: Không có đặc trưng.")
            return {}

        X, y = self.trainer._create_sequences(df, feature_cols, target_col)

        if len(X) < 10:
             logger.warning(f"Bỏ qua {ticker}: Không đủ mẫu ({len(X)}) để tạo chuỗi (cần ít nhất 10).")
             return {}

        try:
            # --- THAY ĐỔI: Sử dụng validation_split thay vì test_split cố định ---
            val_split_ratio = self.config.get('training_quarterly.val_split', 0.2) # Thêm config này
            train_size = int(len(X) * (1 - val_split_ratio))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
             # --- KẾT THÚC THAY ĐỔI ---

        except ValueError:
            logger.warning(f"Không đủ dữ liệu để chia train/val cho {ticker}")
            return {}

        ticker_results = {}

        for model_name, model_class in self.models_to_train.items():
            logger.info(f"--- Bắt đầu {model_name.upper()} hàng quý cho {ticker} ---")

            model_config = self.config.get('models')
            temp_config = model_config.copy()
            temp_config['shared'] = {'forecast_horizons': ['1Q']}

            model = model_class(input_dim=len(feature_cols), config=temp_config)

            # --- THAY ĐỔI: Sử dụng X_val cho Early Stopping ---
            model = self.trainer.train(model, X_train, y_train, X_val, y_val)
            # --- KẾT THÚC THAY ĐỔI ---

            model.eval()
            with torch.no_grad():
                # --- THAY ĐỔI: Đánh giá trên tập Val/Test (X_val) ---
                test_outputs = model(torch.tensor(X_val, dtype=torch.float32).to(self.device))
                y_pred = torch.argmax(test_outputs['Target_Direction_t+1Q'], dim=1).cpu().numpy()
                bal_acc = balanced_accuracy_score(y_val, y_pred) # Đánh giá trên y_val
                # --- KẾT THÚC THAY ĐỔI ---

            ticker_results[model_name] = bal_acc

            # --- THAY ĐỔI: Gọi hàm log mới ---
            self._log_results(ticker, model_name, bal_acc)
            # --- KẾT THÚC THAY ĐỔI ---

            models_dir = self.config.get('paths.models', 'models')
            os.makedirs(models_dir, exist_ok=True) # Đảm bảo thư mục tồn tại
            model_path = os.path.join(models_dir, f"{ticker}_quarterly_{model_name}.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Đã lưu model hàng quý tại: {model_path}")

        return ticker_results

# --- Hàm train_all_quarterly_models (Giữ nguyên logic chính) ---

def train_all_quarterly_models():
    """Hàm chính để chạy huấn luyện hàng quý cho tất cả các mã."""
    print("\n" + "=" * 60)
    print("BƯỚC 3 (NHÁNH 2): HUẤN LUYỆN MODEL HÀNG QUÝ (DL TỪ NHÁNH 1)")
    print("=" * 60)

    config = get_config()
    logger.info(f"Sử dụng thiết bị: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    pipeline = QuarterlyTrainingPipeline()

    processed_dir = config.get('paths.processed', 'data/processed')
    tickers = config.get('data.tickers', [])

    all_results = {model_name: [] for model_name in pipeline.models_to_train.keys()}

    for ticker in tickers:
        file_path = os.path.join(processed_dir, f"{ticker}_quarterly_features.csv")
        if not os.path.exists(file_path):
            continue

        # --- THAY ĐỔI: Đọc index_col='time' ---
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
        # --- KẾT THÚC THAY ĐỔI ---

        ticker_results = pipeline.train_single_ticker(ticker, df)

        for model_name, bal_acc in ticker_results.items():
            all_results[model_name].append(bal_acc)

    print("\n" + "=" * 60)
    print("TỔNG KẾT HUẤN LUYỆN (HÀNG QUÝ)")
    print("=" * 60)

    for model_name, results in all_results.items():
        if results:
            avg_acc = np.mean(results)
            print(f"  - {model_name.upper()} (Trung bình): {avg_acc:.2%}")
        else:
            print(f"  - {model_name.upper()}: Không có kết quả.")
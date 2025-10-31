# src/models/base_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, balanced_accuracy_score
)
import joblib
import matplotlib.pyplot as plt

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger("models")

# --- ĐỊNH NGHĨA LỚP CƠ SỞ ---
class BaseModel(nn.Module, ABC):
    """Lớp cơ sở trừu tượng cho tất cả các mô hình."""
    
    def __init__(self, input_dim: int, config: Dict):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        
        # Lấy các tham số dùng chung từ config
        shared_config = config.get('shared', {})
        
        # horizons giờ đây có thể là ['1', '3', '5'] (Nhánh 1) 
        # hoặc ['1Q'] (Nhánh 2)
        self.horizons = shared_config.get('forecast_horizons', ['1', '3', '5'])
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Phương thức forward bắt buộc phải được triển khai bởi các lớp con.
        Phải trả về một dictionary chứa các tensor đầu ra cho mỗi target.
        """
        pass
# --- KẾT THÚC LỚP CƠ SỞ ---


class ModelTrainer:
    """Lớp trung tâm để điều phối việc huấn luyện model (CHO NHÁNH 1)."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Sử dụng thiết bị: {self.device}")
        self.feature_columns = []
        self.class_weights = {}

    def _calculate_class_weights(self, target_data: np.ndarray) -> torch.Tensor:
        """Tính toán trọng số để xử lý mất cân bằng lớp trong bài toán phân loại."""
        num_classes = self.config.get('models.shared.num_classes', 2)
        class_counts = np.bincount(target_data.astype(int), minlength=num_classes)
        
        total_samples = class_counts.sum()
        if total_samples == 0:
            return torch.ones(num_classes, dtype=torch.float32)

        weights = total_samples / (len(class_counts) * class_counts)
        weights[np.isinf(weights)] = 1.0 # Gán trọng số cơ bản nếu lớp không có mẫu
        
        if weights.mean() > 0:
            weights = weights / weights.mean() # Chuẩn hóa
        weights = np.clip(weights, 0.5, 8.0) # Kẹp giá trị để ổn định
        
        return torch.tensor(weights, dtype=torch.float32)

    def prepare_data_pipeline(self, df: pd.DataFrame) -> Dict:
        """
        Chuẩn bị pipeline từ dataframe đã được scale: chỉ chia và tạo chuỗi.
        """
        
        # Di chuyển logic dropna từ feature_engineer sang đây
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        df = df.dropna(subset=target_cols)
        df = df.reset_index(drop=True)
        
        config = self.config.get('training', {})
        train_split = config.get('train_split', 0.8)
        val_split = config.get('val_split', 0.1)
        timesteps = config.get('timesteps', 30)
        batch_size = config.get('batch_size', 32)
        
        train_size = int(len(df) * train_split)
        val_size = int(len(df) * val_split)

        df_train = df.iloc[:train_size]
        df_val = df.iloc[train_size:train_size + val_size]
        df_test = df.iloc[train_size + val_size:]
        logger.info(f"Chia dữ liệu (sau khi lọc): Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

        metadata_cols = self._load_feature_columns_from_metadata(df['Ticker'].iloc[0])
        if metadata_cols:
             self.feature_columns = metadata_cols
        else:
            numeric_features = df.select_dtypes(include=np.number).columns.tolist()
            self.feature_columns = [col for col in numeric_features if not col.startswith('Target_')]

        if not self.feature_columns:
            raise ValueError("Không tìm thấy đặc trưng số nào để huấn luyện.")

        X_train, y_train_dict = self._create_sequences(df_train, self.feature_columns, timesteps)
        X_val, y_val_dict = self._create_sequences(df_val, self.feature_columns, timesteps)
        X_test, y_test_dict = self._create_sequences(df_test, self.feature_columns, timesteps)
        
        data_loaders = {}
        all_targets_data = {}
        
        for split_name, (X, y_dict) in [('train', (X_train, y_train_dict)), ('val', (X_val, y_val_dict)), ('test', (X_test, y_test_dict))]:
            all_targets_data[split_name] = y_dict
            if X.shape[0] > 0:
                target_tensors = [torch.tensor(y_dict[key], dtype=torch.long if 'Direction' in key else torch.float32) for key in sorted(y_dict.keys())]
                dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), *target_tensors)
                data_loaders[split_name] = DataLoader(dataset, batch_size=batch_size, shuffle=(split_name == 'train'))
        
        return {
            'data_loaders': data_loaders,
            'input_dim': len(self.feature_columns),
            'targets_meta': all_targets_data
        }

    def _load_feature_columns_from_metadata(self, ticker: str) -> List[str]:
        """Tải danh sách đặc trưng từ file metadata."""
        try:
            processed_dir = self.config.get('paths.processed', 'data/processed')
            metadata_path = os.path.join(processed_dir, f"{ticker}_metadata.pkl")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                return metadata.get('feature_columns', [])
        except Exception as e:
            logger.warning(f"Không thể tải metadata cho {ticker}: {e}")
        return []

    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str], timesteps: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Hàm trợ giúp để tạo các chuỗi dữ liệu đầu vào và đầu ra."""
        X, targets = [], {}
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        for col in target_cols:
            targets[col] = []

        if len(df) < timesteps:
            return np.empty((0, timesteps, len(feature_cols))), {k: np.array(v) for k, v in targets.items()}

        valid_feature_cols = [col for col in feature_cols if col in df.columns]
        if len(valid_feature_cols) != len(feature_cols):
             logger.warning(f"Một số cột đặc trưng trong metadata không có trong dataframe. Chỉ sử dụng các cột hợp lệ.")

        for i in range(len(df) - timesteps + 1):
            X.append(df[valid_feature_cols].iloc[i:i + timesteps].values)
            for col in target_cols:
                targets[col].append(df[col].iloc[i + timesteps - 1])

        return np.array(X), {k: np.array(v) for k, v in targets.items()}

    def train_model(self, model: BaseModel, data: Dict) -> Dict:
        """Hàm chính để huấn luyện model với early stopping."""
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 50)
        learning_rate = training_config.get('learning_rate', 0.0001)
        patience = training_config.get('early_stopping_patience', 10)
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_loader = data['data_loaders'].get('train')
        val_loader = data['data_loaders'].get('val')
        if not train_loader or not val_loader:
            logger.error("Không có đủ dữ liệu trong train_loader hoặc val_loader. Dừng huấn luyện.")
            return {}

        train_targets = data['targets_meta']['train']
        for key, target_data in train_targets.items():
            if 'Direction' in key and len(target_data) > 0:
                self.class_weights[key] = self._calculate_class_weights(target_data).to(self.device)

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for batch_data in train_loader:
                X_batch = batch_data[0].to(self.device)
                target_batch = [t.to(self.device) for t in batch_data[1:]]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                loss = self._calculate_combined_loss(outputs, target_batch, sorted(train_targets.keys()))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)

            avg_val_loss, _ = self._evaluate_model(model, val_loader, data['targets_meta']['val'])
            history['val_loss'].append(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping tại epoch {epoch+1}")
                    break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
            
        _, test_metrics = self._evaluate_model(model, data['data_loaders'].get('test'), data['targets_meta']['test'])

        return {'model': model, 'history': history, 'test_metrics': test_metrics}

    def _calculate_combined_loss(self, outputs, targets, target_keys):
        """Tính toán loss tổng hợp cho cả tác vụ hồi quy và phân loại."""
        total_loss = 0
        loss_weights = self.config.get('training.loss_weights', {})
        
        for i, key in enumerate(target_keys):
            if key not in outputs: continue # Bỏ qua nếu model không có đầu ra này
                
            output = outputs[key]
            target = targets[i]
            
            if 'Direction' in key:
                weight = self.class_weights.get(key)
                criterion = nn.CrossEntropyLoss(weight=weight)
                loss = criterion(output, target.long())
                total_loss += loss * loss_weights.get('classification', 1.0)
            else:
                criterion = nn.MSELoss()
                loss = criterion(output.squeeze(), target)
                total_loss += loss * loss_weights.get('regression', 1.0)
                
        return total_loss
        
    def _evaluate_model(self, model, data_loader, targets_meta):
        """Đánh giá hiệu suất của model trên một tập dữ liệu."""
        if not data_loader:
            return 0.0, {}
            
        model.eval()
        total_loss = 0
        predictions = {key: [] for key in targets_meta.keys()}
        
        with torch.no_grad():
            for batch_data in data_loader:
                X_batch = batch_data[0].to(self.device)
                target_batch = [t.to(self.device) for t in batch_data[1:]]
                
                outputs = model(X_batch)
                loss = self._calculate_combined_loss(outputs, target_batch, sorted(targets_meta.keys()))
                total_loss += loss.item()
                
                for i, key in enumerate(sorted(targets_meta.keys())):
                    if key not in outputs: continue
                    if 'Direction' in key:
                        preds = torch.argmax(outputs[key], dim=1)
                        predictions[key].extend(preds.cpu().numpy())
                    else:
                        predictions[key].extend(outputs[key].squeeze().cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(predictions, targets_meta)
        return avg_loss, metrics

    def _calculate_metrics(self, predictions, targets):
        """Tính toán các chỉ số đánh giá."""
        metrics = {}
        for key, pred_values in predictions.items():
            true_values = targets.get(key)
            if true_values is None or len(true_values) == 0:
                continue

            pred_values = np.array(pred_values)
            true_values = np.array(true_values)
            
            if len(pred_values) == 0: continue # Không có dự đoán nào

            valid_indices = np.isfinite(true_values) & np.isfinite(pred_values)
            true_values = true_values[valid_indices]
            pred_values = pred_values[valid_indices]

            if len(true_values) == 0 or len(pred_values) == 0 or len(true_values) != len(pred_values):
                continue

            if 'Direction' in key:
                if len(np.unique(true_values)) < 2:
                     acc = accuracy_score(true_values, pred_values)
                     metrics[key] = {'accuracy': acc, 'balanced_accuracy': acc}
                else:
                    metrics[key] = {
                        'accuracy': accuracy_score(true_values, pred_values),
                        'balanced_accuracy': balanced_accuracy_score(true_values, pred_values)
                    }
            else:
                if len(true_values) < 2: # Không thể tính R2 nếu chỉ có 1 mẫu
                    metrics[key] = {'rmse': np.sqrt(mean_squared_error(true_values, pred_values)), 'r2': 0.0}
                else:
                    metrics[key] = {
                        'rmse': np.sqrt(mean_squared_error(true_values, pred_values)),
                        'r2': r2_score(true_values, pred_values)
                    }
        return metrics

    def save_model(self, model: BaseModel, ticker: str, model_type: str):
        """Lưu model đã huấn luyện."""
        models_dir = self.config.get('paths.models', 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{ticker}_{model_type}_best.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Đã lưu model tại: {model_path}")
    
    def plot_training_history(self, history, ticker, model_type):
        """Vẽ và lưu biểu đồ quá trình huấn luyện."""
        outputs_dir = self.config.get('paths.outputs', 'outputs')
        os.makedirs(outputs_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Training History for {ticker} - {model_type}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(outputs_dir, f"{ticker}_{model_type}_training.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Đã lưu biểu đồ huấn luyện tại: {plot_path}")
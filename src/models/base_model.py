# src/models/base_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
import os
import math
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from sklearn.metrics import f1_score, accuracy_score

class BaseModel(nn.Module):
    """
    Lớp cơ sở cho tất cả các mô hình, xử lý việc lưu/tải,
    và các thành phần chung như Positional Encoding.
    """
    def __init__(self, input_dim: int, config: Dict):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)

    def save(self, path: str):
        """Lưu trọng số mô hình."""
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Tải trọng số mô hình."""
        self.load_state_dict(torch.load(path, map_location=self.device))

class PositionalEncoding(nn.Module):
    """
    Cung cấp thông tin vị trí cho Transformer.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ModelTrainer:
    """
    Class quản lý quy trình huấn luyện, đánh giá và tối ưu hóa.
    """
    def __init__(self, model: nn.Module, config: Dict, ticker: str = "Unknown", horizon: int = 1):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.ticker = ticker
        self.horizon = horizon
        self.logger = logging.getLogger("trainer")

    def _calculate_class_weights(self, target_data: np.ndarray) -> torch.Tensor:
        """
        Sử dụng công thức Balanced chuẩn của Scikit-learn.
        Weight = n_samples / (n_classes * n_samples_j)
        """
        shared_config = self.config.get('shared', {})
        n_classes = int(shared_config.get('num_classes', 3))

        # Đếm số lượng mẫu mỗi lớp
        class_counts = np.bincount(target_data.astype(int), minlength=n_classes)
        total_samples = class_counts.sum()

        class_dist_str = ", ".join([f"Class {i}={count}" for i, count in enumerate(class_counts)])
        self.logger.info(f"📊 Class Distribution: {class_dist_str}")

        # Xử lý trường hợp thiếu dữ liệu hoặc một lớp nào đó không có mẫu
        if total_samples == 0 or 0 in class_counts:
             self.logger.warning(f"⚠️  Dữ liệu bị thiếu hoặc có lớp không tồn tại! Gán trọng số mặc định cho {n_classes} lớp.")
             return torch.ones(n_classes, dtype=torch.float32).to(self.device)

        # Công thức chuẩn
        weights = total_samples / (n_classes * class_counts)
        
        # Chuyển thành tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        weights_dist_str = ", ".join([f"Class {i}={w:.4f}" for i, w in enumerate(weights)])
        self.logger.info(f"⚖️  Balanced Weights (Standard): {weights_dist_str}")
        return weights_tensor

    def _calculate_loss(self, logits, targets, class_weights):
        """
        Tính toán hàm mất mát. Ưu tiên CrossEntropyLoss tiêu chuẩn.
        """
        targets = targets.long()
        
        # Lấy cấu hình focal loss (nếu có)
        use_focal = bool(self.config.get('training', {}).get('use_focal_loss', False))
        
        if use_focal:
            # Focal Loss Implementation
            gamma = float(self.config.get('training', {}).get('focal_gamma', 2.0))
            
            # Tính Cross Entropy không reduce để áp dụng công thức Focal
            ce_loss = nn.functional.cross_entropy(logits, targets, weight=class_weights, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
            return focal_loss
        else:
            # Standard Cross Entropy (Khuyên dùng để debug/ổn định)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            return criterion(logits, targets)

    def train_epoch(self, train_loader, optimizer, class_weights):
        """Huấn luyện 1 epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []   # <--- ĐÃ SỬA: Khởi tạo list rỗng
        all_targets = [] # <--- ĐÃ SỬA: Khởi tạo list rỗng
        
        grad_clip = float(self.config.get('training', {}).get('gradient_clip_norm', 1.0))

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            
            # Lấy logits từ output dictionary
            logits = outputs
            
            loss = self._calculate_loss(logits, y_batch, class_weights)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            optimizer.step()
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        # Tính metrics huấn luyện
        train_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        return avg_loss, train_f1

    def evaluate(self, val_loader, class_weights):
        """Đánh giá trên tập Validation/Test."""
        self.model.eval()
        total_loss = 0
        all_preds = []   # <--- ĐÃ SỬA
        all_targets = [] # <--- ĐÃ SỬA

        if len(val_loader) == 0:
            return {'loss': 0, 'f1': 0, 'accuracy': 0, 'preds': [], 'targets': []}

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                logits = outputs
                
                loss = self._calculate_loss(logits, y_batch, class_weights)
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        acc = accuracy_score(all_targets, all_preds)
        
        metrics = {
            'loss': avg_loss,
            'f1': f1,
            'accuracy': acc,
            'preds': all_preds,
            'targets': all_targets
        }
        return metrics

    def fit(self, train_loader, val_loader):
        """
        Vòng lặp huấn luyện chính.
        """
        train_cfg = self.config.get('training', {})
        epochs = int(train_cfg.get('epochs', 100))
        lr = float(train_cfg.get('learning_rate', 0.001))
        patience = int(train_cfg.get('early_stopping_patience', 20))
        
        # Thiết lập Optimizer
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=float(train_cfg.get('weight_decay_short', 1e-5))
        )

        # Scheduler: Giảm LR nếu Val Loss không giảm
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=float(train_cfg.get('scheduler_factor', 0.5)), 
            patience=int(train_cfg.get('scheduler_patience', 5)),
            min_lr=float(train_cfg.get('scheduler_min_lr', 1e-6)),
            verbose=True
        )

        # Tính toán Class Weights một lần
        all_train_targets = [] # <--- ĐÃ SỬA: Khởi tạo list rỗng
        for _, y in train_loader:
            all_train_targets.extend(y.numpy())
            
        class_weights = self._calculate_class_weights(np.array(all_train_targets))

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        self.logger.info(f"🚀 Bắt đầu huấn luyện: {self.ticker} (LR={lr})")

        for epoch in range(epochs):
            # 1. Train
            train_loss, train_f1 = self.train_epoch(train_loader, optimizer, class_weights)
            
            # 2. Validate
            val_metrics = self.evaluate(val_loader, class_weights)
            val_loss = val_metrics['loss']
            val_f1 = val_metrics['f1']
            
            # Lưu lịch sử
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            
            # 3. Step Scheduler
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")

            # 4. Early Stopping (Dựa trên Val Loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Lưu model tốt nhất
                save_path = os.path.join(self.config.get('paths.models', 'models'), f"{self.ticker}_best.pt")
                self.model.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info(f"🛑 Early stopping tại epoch {epoch+1}")
                    break
        
        # Vẽ biểu đồ sau khi train xong
        self.plot_history(history)
        
        # Load lại best model để đánh giá lần cuối
        best_model_path = os.path.join(self.config.get('paths.models', 'models'), f"{self.ticker}_best.pt")
        if os.path.exists(best_model_path):
             self.model.load(best_model_path)
             
        return history

    def plot_history(self, history):
        """Vẽ biểu đồ Loss và F1."""
        plt.figure(figsize=(12, 5))
        
        # Biểu đồ Loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'{self.ticker} - Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Biểu đồ F1
        plt.subplot(1, 2, 2)
        plt.plot(history['val_f1'], label='Val F1', color='orange')
        plt.title(f'{self.ticker} - Val F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        save_path = os.path.join(self.config.get('paths.outputs', 'outputs'), f"{self.ticker}_history.png")
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"📉 Đã lưu biểu đồ tại: {save_path}")
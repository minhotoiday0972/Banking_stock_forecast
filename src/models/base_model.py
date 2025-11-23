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
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from ..utils.loss_functions import FocalLoss

logger = logging.getLogger(__name__)


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
    Class để đóng gói quy trình huấn luyện và đánh giá model Pytorch.
    """
    def __init__(self, model: BaseModel, config: Dict, ticker: str, horizon: int, class_weights: Optional[torch.Tensor] = None):
        self.model = model
        self.config = config.get('training', {})
        self.ticker = ticker
        self.horizon = horizon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay_long', 0.001)
        )
        
        if self.config.get('use_focal_loss', True):
            alpha = self.class_weights if self.class_weights is not None else None
            self.criterion = FocalLoss(alpha=alpha, gamma=self.config.get('focal_loss_gamma', 2))
        else:
            # Use class weights if provided and not using Focal Loss
            if self.class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            'max', 
            patience=self.config.get('scheduler_patience', 10), 
            factor=self.config.get('scheduler_factor', 0.1),
            min_lr=float(self.config.get('scheduler_min_lr', 1e-6))
        )

        self.epochs = self.config.get('epochs', 100)
        self.patience = self.config.get('early_stopping_patience', 20)
        self.model_save_path = os.path.join(
            config.get('paths', {}).get('models', 'models'),
            f"{ticker}_{model.__class__.__name__.lower()}_t+{horizon}_best.pt"
        )

    def _evaluate(self, data_loader) -> Tuple[float, float, float]:
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        bal_acc = balanced_accuracy_score(all_targets, all_preds)
        return avg_loss, f1, bal_acc

    def fit(self, train_loader, val_loader) -> float:
        best_val_f1 = 0.0
        epochs_no_improve = 0

        logger.info(f"Starting training for {self.ticker} t+{self.horizon} on {self.device} for {self.epochs} epochs.")

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('gradient_clip_norm', 1.0))
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            val_loss, val_f1, val_bal_acc = self._evaluate(val_loader)
            
            self.scheduler.step(val_f1)

            logger.info(
                f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Bal Acc: {val_bal_acc:.4f}"
            )
            
            # Early stopping logic
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.model.save(self.model_save_path)
                epochs_no_improve = 0
                logger.info(f"Validation F1 improved to {best_val_f1:.4f}. Saving best model.")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs with no improvement.")
                break
        
        # Load best model weights back before finishing
        logger.info(f"Training finished. Loading best model with F1: {best_val_f1:.4f}")
        self.model.load(self.model_save_path)
        return best_val_f1
# src/training/trainer.py
import pandas as pd
import os
import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..models.base_model import ModelTrainer 
from ..models.cnn_bilstm import CNNBiLSTM
from ..models.transformer import TransformerModel

logger = get_logger("trainer") # Logger chính

class ModelTrainingPipeline:
    def __init__(self):
        self.config = get_config()
        self.models_to_train = {
            "cnn_bilstm": CNNBiLSTM,
            "transformer": TransformerModel,
        }
        # Lấy danh sách "công việc" từ config
        self.horizons = self.config.get('models.shared.forecast_horizons', [1, 3, 5, 30, 60, 90])
        logger.info(f"Horizons initialized in ModelTrainingPipeline: {self.horizons} (type: {type(self.horizons)})")

    def _log_and_print_metrics(self, ticker: str, model_type: str, horizon: int, test_metrics: Dict):
        """Log metrics chi tiết bao gồm precision, recall, f1, confusion matrix"""
        
        header = f"--- Kết quả Test cho {ticker} - Model: {model_type.upper()} - Horizon: t+{horizon} ---"
        log_message = [f"\n{header}"]
        print(f"\n{header}")

        if not test_metrics:
            msg = "  Không có metrics nào được tạo (huấn luyện có thể đã thất bại)."
            log_message.append(msg)
            print(msg)
            logger.warning("".join(log_message))
            return

        # Metrics chính
        acc = test_metrics.get('accuracy', float('nan'))
        bal_acc = test_metrics.get('balanced_accuracy', float('nan'))
        precision = test_metrics.get('precision', float('nan'))
        recall = test_metrics.get('recall', float('nan'))
        f1 = test_metrics.get('f1', float('nan'))
        
        msg_acc = f"      - Accuracy: {acc:.2%}"
        msg_bal_acc = f"      - Balanced Accuracy: {bal_acc:.2%}"
        msg_precision = f"      - Precision (Weighted): {precision:.2%}"
        msg_recall = f"      - Recall (Weighted): {recall:.2%}"
        msg_f1 = f"      - F1-Score (Weighted): {f1:.2%}"
        
        log_message.extend([msg_acc, msg_bal_acc, msg_precision, msg_recall, msg_f1])
        print(msg_acc)
        print(msg_bal_acc)
        print(msg_precision)
        print(msg_recall)
        print(msg_f1)
        
        # Confusion Matrix
        if 'confusion_matrix' in test_metrics:
            cm = test_metrics.get('confusion_matrix')
            # Đối với đa lớp, chúng ta sẽ log ma trận đầy đủ
            msg_cm = f"      - Confusion Matrix:\n{np.array(cm)}"
            log_message.append(msg_cm)
            print(msg_cm)
        
        logger.info("\n".join(log_message))

    def _load_data_for_horizon(self, ticker: str, horizon: int) -> Optional[Dict]:
        """Tải dữ liệu và metadata cho một ticker và horizon cụ thể"""
        try:
            processed_dir = self.config.get('paths.processed', 'data/processed')
            
            # Tải metadata
            metadata_path = os.path.join(processed_dir, f"{ticker}_metadata_t+{horizon}.pkl")
            if not os.path.exists(metadata_path):
                logger.error(f"Không tìm thấy metadata: {metadata_path}")
                return None
            
            metadata = joblib.load(metadata_path)
            feature_cols = metadata['feature_columns']
            scaler = metadata['scaler']
            
            # Tải dữ liệu đã scaled
            features_path = os.path.join(processed_dir, f"{ticker}_features_scaled.csv")
            if not os.path.exists(features_path):
                logger.error(f"Không tìm thấy file dữ liệu: {features_path}")
                return None
            
            df = pd.read_csv(features_path)
            target_col = f'Target_Direction_t+{horizon}'
            
            if target_col not in df.columns:
                logger.error(f"Không tìm thấy cột target: {target_col}")
                return None
            
            # Loại bỏ các hàng có NaN trong target hoặc features
            df_clean = df.dropna(subset=[target_col] + feature_cols)
            
            if len(df_clean) < 100:
                logger.error(f"Không đủ dữ liệu sau khi loại bỏ NaN: {len(df_clean)} mẫu")
                return None
            
            logger.info(f"Đã tải {len(df_clean)} mẫu cho {ticker} t+{horizon}")
            
            return {
                'df': df_clean,
                'feature_cols': feature_cols,
                'target_col': target_col,
                'scaler': scaler
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu cho {ticker} t+{horizon}: {e}")
            return None
    
    def _create_dataloaders(self, df: pd.DataFrame, feature_cols: List[str], 
                           target_col: str, sequence_length: int = 30) -> Optional[Dict]:
        """Tạo DataLoaders cho train/val/test"""
        try:
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Tạo sequences
            X_seq, y_seq = [], []
            for i in range(len(X) - sequence_length):
                X_seq.append(X[i:i+sequence_length])
                y_seq.append(y[i+sequence_length])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            if len(X_seq) < 100:
                logger.error(f"Không đủ sequences: {len(X_seq)}")
                return None
            
            # Chia train/val/test: 70/15/15
            train_size = int(len(X_seq) * 0.7)
            val_size = int(len(X_seq) * 0.15)
            
            X_train = torch.FloatTensor(X_seq[:train_size])
            y_train = torch.LongTensor(y_seq[:train_size])
            
            X_val = torch.FloatTensor(X_seq[train_size:train_size+val_size])
            y_val = torch.LongTensor(y_seq[train_size:train_size+val_size])
            
            X_test = torch.FloatTensor(X_seq[train_size+val_size:])
            y_test = torch.LongTensor(y_seq[train_size+val_size:])
            
            batch_size = int(self.config.get('training.batch_size', 32))
            
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            logger.info(f"DataLoaders created: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            
            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'input_dim': len(feature_cols)
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo DataLoaders: {e}")
            return None
    
    def _calculate_test_metrics(self, test_loader, model, device) -> Dict:
        """Tính toán các metrics chi tiết trên tập test"""
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Tính các metrics
        cm = confusion_matrix(all_targets, all_preds)
        
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics

    # --- TÁI CẤU TRÚC LỚN: Hàm này giờ là Vòng lặp chính ---
    def train_all_models(self, model_types: List[str] = None, tickers: List[str] = None) -> Dict[str, Dict[str, Any]]:
        if tickers is None:
            tickers = self.config.get('data.tickers', [])
        if model_types is None:
            model_types = list(self.models_to_train.keys())

        overall_results = {} # Lưu kết quả tổng

        for ticker in tickers:
            for model_type in model_types:
                if model_type not in self.models_to_train:
                    logger.warning(f"Không nhận dạng được loại model: '{model_type}'. Bỏ qua.")
                    continue
                
                # --- THAY ĐỔI: Lặp qua từng Horizon ---
                for horizon in self.horizons:
                    model_key = f"{ticker}_{model_type}_t+{horizon}"
                    overall_results[model_key] = None
                    
                    try:
                        logger.info(f"===== BẮT ĐẦU HUẤN LUYỆN: {model_key} =====")
                        
                        # 1. Tải dữ liệu và metadata
                        data_dict = self._load_data_for_horizon(ticker, horizon)
                        if not data_dict:
                            continue
                        
                        # 2. Tạo DataLoaders
                        sequence_length = int(self.config.get('models.shared.sequence_length', 30))
                        dataloaders = self._create_dataloaders(
                            data_dict['df'], 
                            data_dict['feature_cols'],
                            data_dict['target_col'],
                            sequence_length
                        )
                        if not dataloaders:
                            continue
                        
                        # 3. Khởi tạo Model
                        model_class = self.models_to_train[model_type]
                        model = model_class(
                            input_dim=dataloaders['input_dim'],
                            config=self.config.get('models')
                        )
                        
                        # 4. Khởi tạo ModelTrainer và huấn luyện
                        trainer = ModelTrainer(
                            model=model,
                            config=self.config,
                            ticker=ticker,
                            horizon=horizon
                        )
                        
                        # 5. Huấn luyện
                        history = trainer.fit(dataloaders['train_loader'], dataloaders['val_loader'])
                        
                        # 6. Đánh giá trên tập test
                        test_metrics = self._calculate_test_metrics(
                            dataloaders['test_loader'], 
                            model, 
                            trainer.device
                        )
                        
                        # 7. Ghi lại kết quả
                        overall_results[model_key] = test_metrics
                        self._log_and_print_metrics(ticker, model_type, horizon, test_metrics)
                        
                        # 8. Lưu model
                        models_dir = self.config.get('paths.models', 'models')
                        os.makedirs(models_dir, exist_ok=True)
                        model_path = os.path.join(models_dir, f"{ticker}_{model_type}_t+{horizon}_best.pt")
                        model.save(model_path)
                        logger.info(f"Đã lưu model tại: {model_path}")
                    
                    except Exception as e:
                        logger.exception(f"Lỗi nghiêm trọng khi huấn luyện {model_key}: {e}")
                        self._log_and_print_metrics(ticker, model_type, horizon, None)
        
        return overall_results
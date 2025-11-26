# src/training/trainer.py
import pandas as pd
import os
import torch
import joblib
import numpy as np
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score, accuracy_score

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..models.base_model import ModelTrainer 
from ..models.cnn_bilstm import CNNBiLSTM
from ..models.transformer import TransformerModel
from ..models.baselines import get_baseline_model

import logging

logger_instance = get_logger("trainer")

class ModelTrainingPipeline:
    def __init__(self, config_overrides: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.config = get_config(overrides=config_overrides)
        self.logger = logger if logger is not None else logger_instance
        
        # --- Model Configuration ---
        self._available_dl_models = {"cnn_bilstm": CNNBiLSTM, "transformer": TransformerModel}
        models_to_train_config = self.config.get('models_to_train', {})
        
        # Get model names from config, with fallbacks to empty lists
        dl_model_names = models_to_train_config.get('dl_models', [])
        self.baseline_models = models_to_train_config.get('baseline_models', [])
        
        # Create a dictionary of the DL models that are actually requested
        self.dl_models = {name: self._available_dl_models[name] for name in dl_model_names if name in self._available_dl_models}
        
        # Combine all models to be trained
        self.models_to_train = {**self.dl_models, **{k: None for k in self.baseline_models}}
        self.logger.info(f"Models to train: {list(self.models_to_train.keys())}")

        self.horizons = self.config.get('models.shared.forecast_horizons', [1, 3, 5, 30, 60, 90])
        
        # --- MLflow Setup ---
        mlflow_config = self.config.get('mlflow', {})
        self.mlflow_enabled = mlflow_config.get('enabled', True)
        if self.mlflow_enabled:
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'mlruns'))
            experiment_name = mlflow_config.get('experiment_name', 'Banking Stock Prediction')
            experiment = mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow is enabled. Experiment: '{experiment.name}'")

    def _log_params_to_mlflow(self, params: Dict):
        """Ghi lại các tham số vào MLflow nếu được kích hoạt."""
        if self.mlflow_enabled:
            mlflow.log_params(params)

    def _log_metrics_to_mlflow(self, metrics: Dict):
        """Ghi lại các chỉ số vào MLflow nếu được kích hoạt."""
        if self.mlflow_enabled:
            # Tách confusion matrix ra khỏi các metrics khác
            cm = metrics.pop('confusion_matrix', None)
            mlflow.log_metrics(metrics)
            if cm is not None:
                self._log_cm_to_mlflow(cm)

    def _log_cm_to_mlflow(self, cm_list: List[List[int]]):
        """Vẽ và ghi lại confusion matrix dưới dạng artifact."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(np.array(cm_list), annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            # Lưu vào một file tạm và log
            temp_path = "cm.png"
            plt.savefig(temp_path)
            plots_path = self.config.get('mlflow.artifact_paths.plots', 'plots')
            mlflow.log_artifact(temp_path, plots_path)
            plt.close(fig)
            os.remove(temp_path)
        except Exception as e:
            self.logger.error(f"Could not log confusion matrix plot: {e}")

    def _log_and_print_metrics(self, ticker: str, model_type: str, horizon: int, test_metrics: Dict):
        # (Hàm này giữ nguyên, không thay đổi)
        header = f"--- Kết quả Test cho {ticker} - Model: {model_type.upper()} - Horizon: t+{horizon} ---"
        log_message = [f"\n{header}"]
        print(f"\n{header}")
        if not test_metrics:
            msg = "  Không có metrics nào được tạo."
            log_message.append(msg); print(msg); self.logger.warning("".join(log_message))
            return
        acc = test_metrics.get('accuracy', float('nan'))
        bal_acc = test_metrics.get('balanced_accuracy', float('nan'))
        precision = test_metrics.get('precision', float('nan'))
        recall = test_metrics.get('recall', float('nan'))
        f1 = test_metrics.get('f1', float('nan'))
        msgs = [
            f"      - Accuracy: {acc:.2%}",
            f"      - Balanced Accuracy: {bal_acc:.2%}",
            f"      - Precision (Weighted): {precision:.2%}",
            f"      - Recall (Weighted): {recall:.2%}",
            f"      - F1-Score (Weighted): {f1:.2%}"
        ]
        log_message.extend(msgs); print("\n".join(msgs))
        if 'confusion_matrix' in test_metrics:
            cm = test_metrics.get('confusion_matrix')
            msg_cm = f"      - Confusion Matrix:\n{np.array(cm)}"
            log_message.append(msg_cm); print(msg_cm)
        self.logger.info("\n".join(log_message))

    def _load_data_for_horizon(self, ticker: str, horizon: int) -> Optional[Dict]:
        """
        Tải dữ liệu và metadata. Sửa đổi để KHÔNG loại bỏ các hàng có target là NaN,
        nhằm giữ tính liên tục của chuỗi cho việc tạo sequence.
        """
        try:
            processed_dir = self.config.get('paths.processed', 'data/processed')
            metadata_path = os.path.join(processed_dir, f"{ticker}_metadata_t+{horizon}.pkl")
            features_path = os.path.join(processed_dir, f"{ticker}_features_scaled.csv")

            if not os.path.exists(metadata_path) or not os.path.exists(features_path):
                self.logger.error(f"Thiếu file metadata hoặc features cho {ticker} t+{horizon}")
                return None

            metadata = joblib.load(metadata_path)
            df = pd.read_csv(features_path)
            target_col = f'Target_Direction_t+{horizon}'
            
            if target_col not in df.columns:
                self.logger.error(f"Không tìm thấy cột target: {target_col}")
                return None
            
            # CHỈ loại bỏ NaN từ các cột đặc trưng, giữ lại NaN trong cột target
            df_clean = df.dropna(subset=metadata['feature_columns']).reset_index(drop=True)
            
            if len(df_clean) < 100:
                self.logger.error(f"Không đủ dữ liệu sau khi làm sạch features: {len(df_clean)} mẫu")
                return None
            
            self.logger.info(f"Đã tải {len(df_clean)} mẫu cho {ticker} t+{horizon} (trước khi lọc nhiễu)")
            return {
                'df': df_clean,
                'feature_cols': metadata['feature_columns'],
                'target_col': target_col,
                'scaler': metadata['scaler']
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi tải dữ liệu cho {ticker} t+{horizon}: {e}", exc_info=True)
            return None

    def _calculate_sklearn_metrics(self, y_true, y_pred) -> Dict:
        """
        Tính toán metrics cho các mô hình scikit-learn.
        SỬA ĐỔI: Thêm `labels=[0, 1]` để đảm bảo rằng cả hai lớp luôn được
        tính đến, ngay cả khi một lớp không có trong `y_true` hoặc `y_pred` của một batch,
        giúp tránh các cảnh báo và lỗi hình dạng (shape errors).
        """
        # Ensure labels are specified to handle cases where a class is missing in a batch
        labels = [0, 1]
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
            'confusion_matrix': cm.tolist()
        }

    def _prepare_baseline_data(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Optional[Dict]:
        """
        Chuẩn bị dữ liệu phẳng cho baseline, loại bỏ các mẫu nhiễu (target=NaN).
        """
        try:
            # **LOGIC LỌC NHIỄU**
            df_filtered = df.dropna(subset=[target_col])
            
            if len(df_filtered) < 100:
                self.logger.error(f"Không đủ dữ liệu cho baseline model sau khi lọc nhiễu: {len(df_filtered)} mẫu.")
                return None
            
            self.logger.info(f"Chuẩn bị {len(df_filtered)} mẫu cho baseline model (đã lọc nhiễu).")

            X = df_filtered[feature_cols].values
            y = df_filtered[target_col].values
            
            train_split = self.config.get('training.train_split', 0.7)
            val_split = self.config.get('training.val_split', 0.15)
            
            if train_split + val_split > 0.9:
                val_split = 0.9 - train_split
                self.logger.warning(f"Tổng train+val > 0.9, điều chỉnh val_split còn {val_split:.2f}")

            train_size = int(len(X) * train_split)
            val_size = int(len(X) * val_split)
            
            return {
                "X_train": X[:train_size], "y_train": y[:train_size],
                "X_val": X[train_size:train_size+val_size], "y_val": y[train_size:train_size+val_size],
                "X_test": X[train_size+val_size:], "y_test": y[train_size+val_size:]
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi chuẩn bị dữ liệu baseline: {e}", exc_info=True)
            return None

    def _create_dataloaders(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Optional[Dict]:
        """
        Tạo DataLoaders cho mô hình học sâu, với logic loại bỏ mẫu nhiễu (Sample Rejection).
        """
        try:
            sequence_length = self.config.get('models.shared.sequence_length', 30)
            
            X_data = df[feature_cols].values
            y_data = df[target_col].values

            X_seq, y_seq = [], []
            
            # Duyệt qua các điểm dữ liệu để tạo chuỗi
            for i in range(len(X_data) - sequence_length):
                target_for_sequence = y_data[i + sequence_length]
                
                # **LOGIC LỌC NHIỄU (SAMPLE REJECTION)**
                if not np.isnan(target_for_sequence):
                    X_seq.append(X_data[i:i+sequence_length])
                    y_seq.append(target_for_sequence)

            if len(X_seq) < 100:
                self.logger.error(f"Không đủ chuỗi hợp lệ sau khi lọc nhiễu: {len(X_seq)} chuỗi. Cần ít nhất 100.")
                return None
            
            self.logger.info(f"Đã tạo {len(X_seq)} chuỗi hợp lệ sau khi loại bỏ các mẫu nhiễu.")

            X_seq_np, y_seq_np = np.array(X_seq), np.array(y_seq)
            
            train_split = self.config.get('training.train_split', 0.7)
            val_split = self.config.get('training.val_split', 0.15)
            train_size = int(len(X_seq_np) * train_split)
            val_size = int(len(X_seq_np) * val_split)
            
            X_train, y_train = torch.FloatTensor(X_seq_np[:train_size]), torch.LongTensor(y_seq_np[:train_size])
            X_val, y_val = torch.FloatTensor(X_seq_np[train_size:train_size+val_size]), torch.LongTensor(y_seq_np[train_size:train_size+val_size])
            X_test, y_test = torch.FloatTensor(X_seq_np[train_size+val_size:]), torch.LongTensor(y_seq_np[train_size+val_size:])

            # Tính toán trọng số lớp
            class_counts = torch.bincount(y_train)
            if len(class_counts) < 2: 
                class_weights = torch.ones(2, dtype=torch.float32).detach()
                self.logger.warning(f"Chỉ có 1 lớp được tìm thấy trong dữ liệu training. Đặt trọng số lớp là [1,1].")
            else:
                total_samples = class_counts.sum().float()
                class_weights = (total_samples / (len(class_counts) * class_counts.float())).detach()
            
            self.logger.info(f"Số lượng lớp trong dữ liệu training: {class_counts.tolist()}")
            self.logger.info(f"Trọng số lớp được tính toán: {class_weights.tolist()}")

            batch_size = self.config.get('training.batch_size', 32)
            train_ds, val_ds, test_ds = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val), TensorDataset(X_test, y_test)

            self.logger.info(f"DataLoaders đã được tạo: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
            return {
                'train_loader': DataLoader(train_ds, batch_size=batch_size, shuffle=False),
                'val_loader': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
                'test_loader': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
                'input_dim': len(feature_cols),
                'class_weights': class_weights
            }
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo DataLoaders: {e}", exc_info=True); return None

    def _calculate_dl_test_metrics(self, test_loader, model, device) -> Dict:
        """Tính toán metrics trên tập test cho mô hình học sâu."""
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch.to(device))
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_targets.extend(y_batch.numpy())
        return self._calculate_sklearn_metrics(np.array(all_targets), np.array(all_preds))


    def _run_training_session(self, model_type: str, ticker: str, horizon: int, training_fn):
        """
        Runs a full training and evaluation session, handling MLflow logging.
        """
        model_key = f"{ticker}_{model_type}_t+{horizon}"
        self.logger.info(f"===== STARTING PIPELINE FOR: {model_key} =====")

        try:
            # Luôn cho phép lồng nhau, MLflow sẽ tự động tạo run mới nếu chưa có
            with mlflow.start_run(run_name=f"{model_type}-{ticker}-t{horizon}", nested=True) as run:
                # Log parameters
                # Lấy ID của run cha (nếu có) từ tag của Optuna
                parent_run_id = mlflow.active_run().data.tags.get('mlflow.parentRunId')
                params_to_log = {
                    "model_type": model_type, "ticker": ticker, "horizon": horizon,
                    "parent_run_id": parent_run_id or "N/A", # Ghi lại ID của run cha để dễ truy vết
                    **self.config.get('training', {}),
                    **self.config.get('models', {}).get('shared', {}),
                    **self.config.get('models', {}).get(model_type, {})
                }
                self._log_params_to_mlflow(params_to_log)

                # Load data
                data_dict = self._load_data_for_horizon(ticker, horizon)
                if not data_dict:
                    raise ValueError("Failed to load data.")

                # Execute the specific training and evaluation logic
                results = training_fn(data_dict)
                if not results:
                    raise ValueError("Training function returned no results.")

                # Log metrics and model
                self._log_and_print_metrics(ticker, model_type, horizon, results['test_metrics'])
                
                metrics_to_log = results['test_metrics'].copy()
                if 'best_val_f1' in results:
                    metrics_to_log['best_validation_f1'] = results['best_val_f1']
                self._log_metrics_to_mlflow(metrics_to_log)

                if self.mlflow_enabled and 'model' in results:
                    if model_type in self.dl_models:
                        mlflow.pytorch.log_model(results['model'], f"{model_type}_model", registered_model_name=model_key)
                    else:
                        mlflow.sklearn.log_model(results['model'], f"{model_type}_model")

                return results['test_metrics']

        except Exception as e:
            self.logger.exception(f"FATAL ERROR in pipeline {model_key}: {e}")
            self._log_and_print_metrics(ticker, model_type, horizon, None)
            return None

    def _run_baseline_pipeline(self, model_type: str, ticker: str, horizon: int):
        
        def train_baseline(data_dict):
            baseline_data = self._prepare_baseline_data(data_dict['df'], data_dict['feature_cols'], data_dict['target_col'])
            if not baseline_data: return None

            model = get_baseline_model(model_type, self.config)
            self.logger.info(f"Fitting baseline model '{model_type}'...")
            model.fit(baseline_data['X_train'], baseline_data['y_train'])

            # Evaluate on validation set
            self.logger.info("Evaluating baseline model on validation set...")
            val_predictions = model.predict(baseline_data['X_val'])
            val_metrics = self._calculate_sklearn_metrics(baseline_data['y_val'], val_predictions)
            best_val_f1 = val_metrics.get('f1', 0)
            self.logger.info(f"Validation F1 for {model_type}: {best_val_f1:.4f}")

            # Evaluate on test set
            self.logger.info("Evaluating baseline model on test set...")
            test_predictions = model.predict(baseline_data['X_test'])
            test_metrics = self._calculate_sklearn_metrics(baseline_data['y_test'], test_predictions)

            return {"test_metrics": test_metrics, "model": model, "best_val_f1": best_val_f1}

        return self._run_training_session(model_type, ticker, horizon, train_baseline)

    def _run_dl_pipeline(self, model_type: str, ticker: str, horizon: int):

        def train_dl(data_dict):
            dataloaders = self._create_dataloaders(data_dict['df'], data_dict['feature_cols'], data_dict['target_col'])
            if not dataloaders: return None

            model_class = self.dl_models[model_type]
            model = model_class(input_dim=dataloaders['input_dim'], config=self.config.get('models'))
            
            trainer = ModelTrainer(model=model, config=self.config, ticker=ticker, horizon=horizon, class_weights=dataloaders['class_weights'])
            best_val_f1 = trainer.fit(dataloaders['train_loader'], dataloaders['val_loader'])
            
            test_metrics = self._calculate_dl_test_metrics(dataloaders['test_loader'], model, trainer.device)
            
            return {"test_metrics": test_metrics, "model": model, "best_val_f1": best_val_f1}

        return self._run_training_session(model_type, ticker, horizon, train_dl)

    def train_all_models(self, model_types: List[str] = None, tickers: List[str] = None) -> Dict[str, Dict[str, Any]]:
        tickers = tickers or self.config.get('data.tickers', [])
        model_types = model_types or list(self.models_to_train.keys())
        overall_results = {}

        for ticker in tickers:
            for model_type in model_types:
                if model_type not in self.models_to_train:
                    self.logger.warning(f"Unrecognized model type: '{model_type}'. Skipping.")
                    continue
                for horizon in self.horizons:
                    model_key = f"{ticker}_{model_type}_t+{horizon}"
                    
                    if model_type in self.dl_models:
                        results = self._run_dl_pipeline(model_type, ticker, horizon)
                    elif model_type in self.baseline_models:
                        results = self._run_baseline_pipeline(model_type, ticker, horizon)
                    else:
                        self.logger.error(f"Invalid model configuration for '{model_type}'.")
                        results = None
                    
                    overall_results[model_key] = results
        return overall_results
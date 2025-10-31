# src/training/trainer.py
import pandas as pd
import os
from typing import Dict, List, Any

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..models.base_model import ModelTrainer
from ..models.cnn_bilstm import CNNBiLSTM
from ..models.transformer import TransformerModel

logger = get_logger("trainer")

class ModelTrainingPipeline:
    def __init__(self):
        self.config = get_config()
        self.models_to_train = {
            "cnn_bilstm": CNNBiLSTM,
            "transformer": TransformerModel,
        }

    def train_single_model(self, ticker: str, model_name: str) -> Dict[str, Any]:
        logger.info(f"--- Bắt đầu huấn luyện {model_name.upper()} cho {ticker} ---")
        trainer = ModelTrainer(model_name)

        # Đọc file đặc trưng đã được scale từ bước trước
        processed_dir = self.config.get('paths.processed', 'data/processed')
        features_path = os.path.join(processed_dir, f"{ticker}_features_scaled.csv")
        
        # SỬA LỖI: Kiểm tra file metadata thay vì file scalers cũ
        metadata_path = os.path.join(processed_dir, f"{ticker}_metadata.pkl")
        if not os.path.exists(features_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Không tìm thấy file features_scaled hoặc metadata cho {ticker}. Vui lòng chạy lại 'python main.py features'.")
        
        df = pd.read_csv(features_path, parse_dates=['time'])

        # Chuẩn bị pipeline dữ liệu (chỉ chia và tạo chuỗi, không scale)
        data_pipeline_output = trainer.prepare_data_pipeline(df)

        # Khởi tạo model (sử dụng config đầy đủ)
        model_class = self.models_to_train[model_name]
        model_config = self.config.get('models') # Lấy toàn bộ config models
        
        model = model_class(data_pipeline_output['input_dim'], model_config)

        # Huấn luyện model
        training_results = trainer.train_model(model, data_pipeline_output)
        
        if not training_results:
            logger.error(f"Huấn luyện thất bại cho {ticker} với model {model_name}")
            return None

        # Lưu model và plots
        trainer.save_model(training_results['model'], ticker, model_name)
        if training_results.get('history'):
            trainer.plot_training_history(training_results['history'], ticker, model_name.upper())

        logger.info(f"--- Hoàn thành huấn luyện {model_name.upper()} cho {ticker} ---")
        return training_results

    # --- HÀM MỚI ĐƯỢC THÊM ---
    def _log_and_print_metrics(self, ticker: str, model_type: str, test_metrics: Dict):
        """Hàm trợ giúp để định dạng, in và lưu log kết quả test."""
        
        header = f"--- Kết quả Test cho {ticker} - Model: {model_type.upper()} ---"
        log_message = [f"\n{header}"]
        print(f"\n{header}") # In ra console ngay lập tức

        if not test_metrics:
            msg = "  Không có metrics nào được tạo (huấn luyện có thể đã thất bại)."
            log_message.append(msg)
            print(msg)
            logger.warning("".join(log_message)) # Ghi log là warning
            return

        sorted_targets = sorted(test_metrics.keys())
        
        for target_name in sorted_targets:
            metrics = test_metrics[target_name]
            if not metrics: 
                log_message.append(f"    - {target_name}: Không có metrics.")
                print(f"    - {target_name}: Không có metrics.")
                continue

            log_message.append(f"    - {target_name}:")
            print(f"    - {target_name}:")
            
            if 'rmse' in metrics:
                rmse = metrics.get('rmse', float('nan'))
                r2 = metrics.get('r2', float('nan'))
                msg_rmse = f"      - RMSE: {rmse:.4f}  (Sai số trung bình)"
                msg_r2 =   f"      - R²:   {r2:.4f}  (Mức độ giải thích)"
                log_message.append(msg_rmse)
                log_message.append(msg_r2)
                print(msg_rmse)
                print(msg_r2)
            
            elif 'accuracy' in metrics:
                acc = metrics.get('accuracy', float('nan'))
                bal_acc = metrics.get('balanced_accuracy', float('nan'))
                msg_acc = f"      - Accuracy: {acc:.2%} (Độ chính xác tổng thể)"
                msg_bal_acc = f"      - Balanced Accuracy: {bal_acc:.2%} (Độ chính xác trên các lớp mất cân bằng)"
                log_message.append(msg_acc)
                log_message.append(msg_bal_acc)
                print(msg_acc)
                print(msg_bal_acc)
        
        # Ghi toàn bộ thông báo vào log file
        logger.info("\n".join(log_message))
    # --- KẾT THÚC HÀM MỚI ---

    def train_all_models(self, model_types: List[str] = None, tickers: List[str] = None) -> Dict[str, Dict[str, Any]]:
        if tickers is None:
            tickers = self.config.get('data.tickers', [])
        if model_types is None:
            model_types = list(self.models_to_train.keys())

        overall_results = {model_type: {} for model_type in model_types}

        for model_type in model_types:
            if model_type not in self.models_to_train:
                logger.warning(f"Không nhận dạng được loại model: '{model_type}'. Bỏ qua.")
                continue
            
            logger.info(f"===== ĐANG HUẤN LUYỆN MODEL: {model_type.upper()} =====")
            for ticker in tickers:
                try:
                    result = self.train_single_model(ticker, model_type)
                    overall_results[model_type][ticker] = result
                    
                    # --- THAY ĐỔI: In và log kết quả ngay lập tức ---
                    if result and result.get('test_metrics'):
                        self._log_and_print_metrics(ticker, model_type, result['test_metrics'])
                    else:
                        # Ghi lại thất bại
                        self._log_and_print_metrics(ticker, model_type, None)
                    # --- KẾT THÚC THAY ĐỔI ---

                except Exception as e:
                    logger.exception(f"Lỗi nghiêm trọng khi huấn luyện {model_type} cho {ticker}: {e}")
                    overall_results[model_type][ticker] = None
                    self._log_and_print_metrics(ticker, model_type, None) # Ghi lại thất bại do exception
        
        return overall_results
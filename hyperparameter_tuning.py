# hyperparameter_tuning.py
import argparse
import optuna
from optuna.integration import MLflowCallback
import logging
import sys
import mlflow

# Cấu hình logging cơ bản để hiển thị tất cả các log từ INFO trở lên
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Thêm src vào sys.path để có thể import từ các module trong src
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.trainer import ModelTrainingPipeline
from src.utils.logger import get_logger

# Thiết lập logger cho kịch bản này
logger = get_logger("HyperTuning", log_filename="hyperparameter_tuning.log")
# Tắt bớt log từ Optuna để không bị nhiễu
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, model_type: str, ticker: str, horizon: int) -> float:
    """
    Hàm mục tiêu cho Optuna study. Optuna sẽ cố gắng tối đa hóa giá trị trả về của hàm này.
    """
    # Bắt đầu một MLflow run lồng nhau để có thể đặt tên tùy chỉnh
    with mlflow.start_run(nested=True) as run:
        # Đặt tên cho run này thông qua tag, đây là cách làm chuẩn
        run_name = f"hptune-{model_type}-{ticker}-t{horizon}-trial-{trial.number}"
        mlflow.set_tag("mlflow.runName", run_name)
        logger.info(f"--- Trial #{trial.number} | MLflow Run: {run_name} ---")

        # 1. Định nghĩa không gian tìm kiếm cho các siêu tham số
        if model_type == 'cnn_bilstm':
            overrides = {
                'training': {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                },
                'models': {
                    'cnn_bilstm': {
                        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
                        'num_layers': trial.suggest_int('num_layers', 1, 2),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
                    }
                }
            }
        elif model_type == 'transformer':
            overrides = {
                'training': {
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                },
                'models': {
                    'transformer': {
                        'd_model': trial.suggest_categorical('d_model', [32, 64]),
                        'nhead': trial.suggest_categorical('nhead', [2, 4]),
                        'num_layers': trial.suggest_int('num_layers', 1, 2),
                        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),
                    }
                }
            }
        else:
            raise ValueError(f"Model type '{model_type}' không được hỗ trợ để tối ưu siêu tham số.")

        try:
            # 2. Khởi tạo và chạy pipeline với cấu hình ghi đè
            pipeline = ModelTrainingPipeline(config_overrides=overrides)
            
            # Chỉ chạy một quy trình duy nhất
            if model_type in pipeline.dl_models:
                results = pipeline._run_dl_pipeline(model_type, ticker, horizon)
            else:
                logger.warning(f"Bỏ qua tối ưu cho model không phải DL: {model_type}")
                return 0.0

            if results is None:
                logger.error("Quy trình huấn luyện thất bại, không có kết quả.")
                return 0.0 # Trả về điểm thấp nếu lần chạy thất bại

            # 3. Trả về điểm validation F1 để Optuna tối ưu
            val_f1 = results.get('best_val_f1', 0.0)
            logger.info(f"--- Trial #{trial.number} Finished. Validation F1: {val_f1:.4f} ---")
            return val_f1

        except Exception as e:
            logger.exception(f"Một lỗi nghiêm trọng xảy ra trong trial: {e}")
            # Trả về điểm thấp để Optuna biết lần thử này thất bại
            return 0.0


def main():
    parser = argparse.ArgumentParser(description="Tối ưu siêu tham số với Optuna và MLflow")
    parser.add_argument('--model', type=str, required=True, choices=['cnn_bilstm', 'transformer'], help='Model cần tối ưu.')
    parser.add_argument('--ticker', type=str, required=True, help='Mã cổ phiếu (VD: VCB).')
    parser.add_argument('--horizon', type=int, required=True, help='Chân trời dự báo (VD: 5).')
    parser.add_argument('--trials', type=int, default=50, help='Số lần thử.')
    parser.add_argument('--study-name', type=str, default=None, help='Tên của Optuna study.')
    args = parser.parse_args()

    # MLflow callback để tự động log các trial của Optuna.
    # Nó sẽ tự động gắn vào run đang hoạt động được tạo bên trong hàm objective.
    mlflow_callback = MLflowCallback(
        tracking_uri="mlruns",
        metric_name="best_validation_f1"
    )

    study_name = args.study_name or f"study-{args.model}-{args.ticker}-t{args.horizon}"
    
    # Sử dụng lambda để truyền thêm tham số vào hàm objective
    objective_func = lambda trial: objective(trial, args.model, args.ticker, args.horizon)

    logger.info(f"Bắt đầu Optuna study '{study_name}' với {args.trials} trials...")
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage='sqlite:///optuna_studies.db', # Lưu kết quả vào database
        load_if_exists=True # Có thể tiếp tục study cũ nếu có
    )
    
    study.optimize(objective_func, n_trials=args.trials, callbacks=[mlflow_callback])

    logger.info("Hoàn tất tối ưu siêu tham số.")
    logger.info(f"Trial tốt nhất cho study '{study_name}':")
    logger.info(f"  - Value (Best Val F1): {study.best_trial.value:.4f}")
    logger.info("  - Params: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    - {key}: {value}")

if __name__ == "__main__":
    main()
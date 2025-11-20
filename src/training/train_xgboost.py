# File: train_xgboost.py
# ---------------------
# Script này CHỈ huấn luyện mô hình XGBoost cho các
# horizon dài hạn (t+30, t+60, t+90) để so sánh hiệu suất
# và kiểm tra vấn đề overfitting.
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from typing import List, Dict, Any, Tuple

# --- TÙY CHỈNH IMPORT ---
# Đảm bảo các đường dẫn này khớp với cấu trúc dự án của Sir
try:
    # (Chúng ta không cần ModelTrainer hay BankingDataModule ở đây)
    from src.utils.logger import get_logger
    from src.utils.config import get_config
except ImportError as e:
    print(f"Lỗi Import: {e}")
    print("Vui lòng đảm bảo các đường dẫn import (from src...) là chính xác.")
    exit(1)
# -------------------------

logger = get_logger("xgb_trainer")
config = get_config()

# === CÀI ĐẶT HUẤN LUYỆN ===
TICKERS_TO_RUN = config.get('data.tickers', ['ACB', 'VIB']) 

# === CHỈ CHẠY DÀI HẠN ===
# Chúng ta đã chứng minh ngắn hạn là 50/50, không cần test lại
HORIZONS_TO_TEST = [30, 60, 90]

# === CÀI ĐẶT CHIA DỮ LIỆU (Giống như file train.py) ===
TEST_RATIO = 0.1 
VAL_RATIO = 0.1 
# ---------------------------------------------

def load_full_data_and_metadata(ticker: str, horizon: int) -> (pd.DataFrame, Dict[str, Any]):
    """
    Tải file features_scaled.csv và metadata cho một horizon cụ thể.
    (Hàm này sao chép từ train.py)
    """
    processed_dir = config.get('paths.processed', 'data/processed')
    features_path = os.path.join(processed_dir, f"{ticker}_features_scaled.csv")
    metadata_path = os.path.join(processed_dir, f"{ticker}_metadata_t+{horizon}.pkl")
    
    if not os.path.exists(features_path) or not os.path.exists(metadata_path):
        logger.error(f"Không tìm thấy file data/metadata cho {ticker} H={horizon}")
        return None, None

    logger.debug(f"Đang tải data từ {features_path}")
    df = pd.read_csv(features_path)
    logger.debug(f"Đang tải metadata từ {metadata_path}")
    metadata = joblib.load(metadata_path)
    
    return df, metadata

def get_static_splits(df_length: int, test_ratio: float, val_ratio: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Chia dữ liệu thành các mảng index Train, Val, Test cố định (tuần tự)
    (Hàm này sao chép từ train.py)
    """
    test_size = int(df_length * test_ratio)
    train_val_size = df_length - test_size
    val_size = int(train_val_size * val_ratio)
    train_size = train_val_size - val_size
    
    train_idx = np.arange(0, train_size)
    val_idx = np.arange(train_size, train_size + val_size)
    test_idx = np.arange(train_size + val_size, df_length)
    
    logger.info(f"Chia dữ liệu: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    return train_idx, val_idx, test_idx

def run_xgb_training():
    """
    Hàm chính điều khiển huấn luyện XGBoost.
    """
    model_save_dir = config.get('paths.models', 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Nơi lưu kết quả cuối cùng
    all_results = []

    for ticker in TICKERS_TO_RUN:
        logger.info(f"===== BẮT ĐẦU XỬ LÝ XGBOOST CHO MÃ: {ticker} =====")
        
        for horizon in HORIZONS_TO_TEST:
            model_full_name = f"{ticker}_xgboost_t+{horizon}"
            target_col = f'Target_Direction_t+{horizon}'
            
            logger.info(f"--- Đang tải dữ liệu cho {model_full_name} ---")
            df_full, metadata = load_full_data_and_metadata(ticker, horizon)
            if df_full is None:
                logger.warning(f"Bỏ qua {model_full_name} do thiếu dữ liệu.")
                continue
                
            feature_cols = metadata['feature_columns']
            
            # 2. Làm sạch và Chuẩn bị dữ liệu
            original_len = len(df_full)
            df_full = df_full.dropna(subset=[target_col] + feature_cols)
            df_full = df_full.reset_index(drop=True)
            cleaned_len = len(df_full)
            logger.debug(f"Đã làm sạch {original_len - cleaned_len} hàng NaN, còn lại {cleaned_len} mẫu.")

            if cleaned_len < 200:
                 logger.warning(f"Dữ liệu quá ít ({cleaned_len} mẫu) cho {model_full_name}. Bỏ qua.")
                 continue
                 
            # 3. Chia Train/Val/Test
            train_idx, val_idx, test_idx = get_static_splits(cleaned_len, TEST_RATIO, VAL_RATIO)
            
            # 4. Chuẩn bị data (XGBoost không cần DataModule hay Windowing)
            # Nó đọc thẳng NumPy/Pandas
            X_train = df_full.loc[train_idx, feature_cols]
            y_train = df_full.loc[train_idx, target_col]
            
            X_val = df_full.loc[val_idx, feature_cols]
            y_val = df_full.loc[val_idx, target_col]
            
            X_test = df_full.loc[test_idx, feature_cols]
            y_test = df_full.loc[test_idx, target_col]
            
            # 5. Khởi tạo mô hình XGBoost
            model = xgb.XGBClassifier(
                n_estimators=1000,         # Số lượng cây tối đa (lớn)
                learning_rate=0.01,        # Tốc độ học (nhỏ)
                objective='binary:logistic', # Bài toán phân loại nhị phân
                eval_metric='logloss',       # Metric để early stopping
                early_stopping_rounds=30,  # Dừng lại nếu 30 vòng liên tiếp không cải thiện
                n_jobs=-1,                 # Dùng tất cả CPU
                random_state=42,
                use_label_encoder=False    # Tắt cảnh báo
            )
            
            # 6. Huấn luyện mô hình
            logger.info(f"===== BẮT ĐẦU HUẤN LUYỆN: {model_full_name} =====")
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)], # Dùng tập Val để Early Stopping
                verbose=False # Tắt log của từng epoch
            )
            
            logger.info(f"Huấn luyện hoàn tất. Số cây tốt nhất: {model.best_iteration}")

            # 7. Đánh giá trên tập Test
            preds = model.predict(X_test)
            
            acc = accuracy_score(y_test, preds) * 100
            bacc = balanced_accuracy_score(y_test, preds) * 100
            
            # 8. In kết quả (theo format quen thuộc)
            logger.info(f"--- Kết quả Test cho {ticker} - Model: XGBOOST - Horizon: t+{horizon} ---")
            logger.info(f"    - Accuracy: {acc:.2f}% (Độ chính xác tổng thể)")
            logger.info(f"    - Balanced Accuracy: {bacc:.2f}% (Độ chính xác trên các lớp mất cân bằng)")
            
            # 9. Lưu model
            model_save_path = os.path.join(model_save_dir, f"{model_full_name}_best.pkl")
            joblib.dump(model, model_save_path)
            logger.info(f"Đã lưu model tại: {model_save_path}\n")
            
            all_results.append({
                "Ticker": ticker,
                "Model": "XGBOOST",
                "Horizon": f"t+{horizon}",
                "Balanced Accuracy": bacc,
                "Accuracy": acc
            })

    # In bảng tổng kết cuối cùng
    logger.info("===== TỔNG KẾT HUẤN LUYỆN XGBOOST =====")
    results_df = pd.DataFrame(all_results)
    
    if not results_df.empty:
        with pd.option_context('display.max_rows', None, 'display.width', 1000):
            print(results_df.sort_values(by=['Ticker', 'Horizon']))
        
        # Lưu file tổng kết
        summary_path = os.path.join(config.get('paths.outputs', 'outputs'), 'xgboost_summary_results.csv')
        results_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        logger.info(f"Đã lưu file tổng kết tại: {summary_path}")

if __name__ == "__main__":
    logger.info("===== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN XGBOOST (CHỈ DÀI HẠN) =====")
    
    # Ghi lại config cơ bản
    logger.info(f"TICKER(S): {TICKERS_TO_RUN}")
    logger.info(f"HORIZON(S): {HORIZONS_TO_TEST}")
    
    run_xgb_training()
    
    logger.info("===== QUY TRÌNH HUẤN LUYỆN XGBOOST HOÀN TẤT =====")
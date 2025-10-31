# src/training/train_baseline_rf.py
"""
Huấn luyện mô hình Baseline (RandomForest) cho Nhánh 1 (Dự đoán Hàng ngày).
Sử dụng cùng bộ đặc trưng và cách chia dữ liệu như mô hình Deep Learning.
"""
import pandas as pd
import numpy as np
import os
import joblib
import sys
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split # Sử dụng split đơn giản hơn cho baseline

# Thêm thư mục gốc vào path để import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("train_baseline_rf")

def train_rf_baseline(ticker: str, target_col: str):
    """
    Huấn luyện RandomForest trên dữ liệu hàng ngày cho một mã ticker.
    """
    print("\n" + "=" * 60)
    print(f"HUẤN LUYỆN BASELINE RANDOMFOREST (NHÁNH 1)")
    print(f"Ticker: {ticker} | Mục tiêu: {target_col}")
    print("=" * 60)

    try:
        # 1. Tải Config và các đường dẫn
        config = get_config()
        processed_dir = config.get('paths.processed', 'data/processed')
        models_dir = config.get('paths.models', 'models')
        os.makedirs(models_dir, exist_ok=True)

        # 2. Tải Metadata (để lấy danh sách đặc trưng "vàng")
        metadata_path = os.path.join(processed_dir, f"{ticker}_metadata.pkl")
        if not os.path.exists(metadata_path):
            logger.error(f"Không tìm thấy file metadata: {metadata_path}")
            print(f"Lỗi: Không tìm thấy file metadata cho {ticker}. Chạy 'python main.py features' trước.")
            return
        metadata = joblib.load(metadata_path)
        feature_cols = metadata.get('feature_columns')
        if not feature_cols:
            logger.error("Không tìm thấy 'feature_columns' trong metadata.")
            print("Lỗi: Không tìm thấy danh sách đặc trưng trong metadata.")
            return

        logger.info(f"Sử dụng {len(feature_cols)} đặc trưng đã lọc từ metadata.")

        # 3. Tải dữ liệu đã scale
        data_path = os.path.join(processed_dir, f"{ticker}_features_scaled.csv")
        if not os.path.exists(data_path):
            logger.error(f"Không tìm thấy file dữ liệu: {data_path}")
            print(f"Lỗi: Không tìm thấy file dữ liệu {ticker}_features_scaled.csv.")
            return
        df = pd.read_csv(data_path, parse_dates=['time'])
        df = df.sort_values('time') # Đảm bảo dữ liệu theo thứ tự thời gian

        # 4. Chuẩn bị dữ liệu
        # Loại bỏ các hàng không có mục tiêu (ví dụ 5 hàng cuối)
        df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)
        if df_clean.empty:
            logger.error(f"Không có dữ liệu hợp lệ cho mục tiêu '{target_col}' sau khi dropna.")
            print(f"Lỗi: Không có dữ liệu hợp lệ cho mục tiêu {target_col}.")
            return

        X = df_clean[feature_cols]
        y = df_clean[target_col].astype(int)

        # 5. Chia dữ liệu Train/Test (Tương tự Nhánh 1 Deep Learning)
        # Lấy tỷ lệ chia từ config của Nhánh 1
        train_cfg = config.get('training', {})
        train_split_ratio = train_cfg.get('train_split', 0.8)
        val_split_ratio = train_cfg.get('val_split', 0.1) # Val set sẽ không dùng, gộp vào test

        # Chia theo thứ tự thời gian
        train_size = int(len(X) * train_split_ratio)
        # Test set sẽ là phần còn lại (bao gồm cả val set của DL)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        if len(X_train) == 0 or len(X_test) == 0:
             logger.error(f"Không đủ dữ liệu để chia train/test (Train: {len(X_train)}, Test: {len(X_test)}).")
             print("Lỗi: Không đủ dữ liệu để chia train/test.")
             return

        logger.info(f"Chia dữ liệu: {len(X_train)} mẫu Huấn luyện, {len(X_test)} mẫu Kiểm tra.")

        # 6. Huấn luyện RandomForest
        logger.info("Bắt đầu huấn luyện RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=200,          # Tăng số cây cho ổn định
            random_state=42,
            n_jobs=-1,                 # Sử dụng tất cả CPU cores
            class_weight='balanced',   # Quan trọng: Xử lý mất cân bằng
            max_depth=15,              # Giới hạn độ sâu để tránh overfitting
            min_samples_split=10,      # Yêu cầu ít nhất 10 mẫu để chia nhánh
            min_samples_leaf=5         # Yêu cầu ít nhất 5 mẫu ở mỗi lá
        )
        model.fit(X_train, y_train)
        logger.info("Huấn luyện hoàn tất.")

        # 7. Đánh giá trên tập Test
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        print("\n" + "=" * 60)
        print(f"KẾT QUẢ BASELINE RANDOMFOREST CHO {ticker} (TRÊN TẬP TEST)")
        print(f"Mục tiêu: {target_col}")
        print("=" * 60)
        print(f"  Accuracy:          {acc:.2%}")
        print(f"  Balanced Accuracy: {bal_acc:.2%}")
        print("\nChi tiết báo cáo phân loại:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Log kết quả chính
        logger.info(f"Kết quả RF Baseline {ticker} ({target_col}): Acc={acc:.2%}, Bal_Acc={bal_acc:.2%}")

        # 8. Lưu model
        model_filename = f"{ticker}_baseline_rf_{target_col.split('+')[-1]}d.pkl" # Ví dụ: ACB_baseline_rf_5d.pkl
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        logger.info(f"Đã lưu model baseline tại: {model_path}")

    except FileNotFoundError as e:
         print(f"Lỗi: {e}. Vui lòng đảm bảo đã chạy bước xử lý đặc trưng ('features') trước.")
         logger.error(f"FileNotFoundError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Đã xảy ra lỗi không mong muốn: {e}", exc_info=True)
        print(f"Đã xảy ra lỗi: {e}")

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Baseline RandomForest cho Nhánh 1")
    parser.add_argument("--ticker", type=str, required=True, help="Mã ticker để huấn luyện (ví dụ: ACB)")
    parser.add_argument(
        "--target",
        type=str,
        default="Target_Direction_t+5",
        choices=["Target_Direction_t+1", "Target_Direction_t+3", "Target_Direction_t+5"],
        help="Cột mục tiêu dự đoán (t+1, t+3, t+5)"
    )
    # Thêm đối số để huấn luyện tất cả tickers
    parser.add_argument("--all_tickers", action='store_true', help="Huấn luyện cho tất cả các ticker trong config")

    args = parser.parse_args()

    if args.all_tickers:
        config = get_config()
        tickers = config.get('data.tickers', [])
        if not tickers:
            print("Lỗi: Không tìm thấy danh sách tickers trong config.yaml")
            return
        logger.info(f"Bắt đầu huấn luyện baseline cho tất cả tickers: {tickers}")
        for ticker in tickers:
            train_rf_baseline(ticker, args.target)
        logger.info("Hoàn tất huấn luyện baseline cho tất cả tickers.")
    elif args.ticker:
        train_rf_baseline(args.ticker, args.target)
    else:
        print("Lỗi: Cần chỉ định --ticker hoặc --all_tickers")

if __name__ == "__main__":
    main()
# evaluate_features.py
"""
Script để đánh giá độ quan trọng của các đặc trưng (Feature Importance)
sử dụng mô hình RandomForest.

Cách sử dụng:
python evaluate_features.py --ticker ACB --target Target_Direction_t+5
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Thêm thư mục src vào path
sys.path.append(os.path.dirname(__file__))
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger("feature_eval")

def evaluate_features(ticker: str, target_col: str):
    """
    Huấn luyện RandomForest để tìm ra các đặc trưng quan trọng nhất.
    """
    print("\n" + "=" * 60)
    print(f"ĐÁNH GIÁ ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG")
    print(f"Ticker: {ticker} | Mục tiêu: {target_col}")
    print("=" * 60)

    try:
        # 1. Tải Config và các đường dẫn
        config = get_config()
        processed_dir = config.get('paths.processed', 'data/processed')
        
        # 2. Tải Metadata (để lấy danh sách đặc trưng)
        metadata_path = os.path.join(processed_dir, f"{ticker}_metadata.pkl")
        if not os.path.exists(metadata_path):
            logger.error(f"Không tìm thấy file metadata: {metadata_path}")
            return
        metadata = joblib.load(metadata_path)
        feature_cols = metadata.get('feature_columns')
        if not feature_cols:
            logger.error("Không tìm thấy 'feature_columns' trong metadata.")
            return
            
        logger.info(f"Đã tải {len(feature_cols)} đặc trưng từ metadata.")

        # 3. Tải dữ liệu đã scale
        data_path = os.path.join(processed_dir, f"{ticker}_features_scaled.csv")
        if not os.path.exists(data_path):
            logger.error(f"Không tìm thấy file dữ liệu: {data_path}")
            return
        df = pd.read_csv(data_path)

        # 4. Chuẩn bị dữ liệu
        # Đây là bước quan trọng: Loại bỏ các hàng không có mục tiêu (ví dụ 5 hàng cuối)
        df_clean = df.dropna(subset=[target_col])
        if df_clean.empty:
            logger.error(f"Không có dữ liệu hợp lệ cho mục tiêu '{target_col}'.")
            return

        X = df_clean[feature_cols]
        y = df_clean[target_col].astype(int) # Đảm bảo mục tiêu là số nguyên (0 hoặc 1)

        logger.info(f"Sẵn sàng huấn luyện trên {len(X)} mẫu.")

        # 5. Huấn luyện RandomForest
        logger.info("Bắt đầu huấn luyện RandomForest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        logger.info("Huấn luyện hoàn tất.")

        # 6. Đánh giá nhanh
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        print("\n--- Hiệu suất (Trên tập huấn luyện) ---")
        print(f"  Accuracy:          {acc:.2%}")
        print(f"  Balanced Accuracy: {bal_acc:.2%}")
        print("  (Chỉ số này cho thấy model có học được hay không)\n")


        # 7. Lấy và In kết quả
        importances = model.feature_importances_
        results = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

        print("--- KẾT QUẢ FEATURE IMPORTANCE ---")
        print("(Từ cao đến thấp)\n")
        
        for i, (feature, importance) in enumerate(results):
            print(f"  {i+1:2d}. {feature:<30} | Importance: {importance:.6f}")

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Đánh giá Feature Importance")
    parser.add_argument("--ticker", type=str, required=True, help="Mã ticker để đánh giá (ví dụ: ACB)")
    parser.add_argument(
        "--target", 
        type=str, 
        default="Target_Direction_t+5", 
        help="Cột mục tiêu để dự đoán (ví dụ: Target_Direction_t+1)"
    )
    args = parser.parse_args()
    
    evaluate_features(args.ticker, args.target)

if __name__ == "__main__":
    main()
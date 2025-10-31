# src/features/feature_engineer_quarterly.py
"""
Nhánh 2: Xử lý đặc trưng HÀNG QUÝ (Phiên bản cho Deep Learning)
Tạo ra các file dữ liệu hàng quý riêng biệt cho từng mã cổ phiếu.
"""

import pandas as pd
import numpy as np
import os
import joblib
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_database

logger = get_logger("feature_engineer_quarterly")

class FeatureEngineerQuarterly:
    def __init__(self):
        self.config = get_config()
        self.db = get_database()

    def create_quarterly_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo mục tiêu dự đoán xu hướng cho quý tiếp theo (t+1Q)."""
        df = df.copy()
        df['Quarterly_Return'] = df['Close'].pct_change()
        df['Target_Return_t+1Q'] = df['Quarterly_Return'].shift(-1)
        # Sửa đổi tên mục tiêu để khớp với logic model
        df['Target_Direction_t+1Q'] = np.where(df['Target_Return_t+1Q'] > 0.0, 1, 0)
        return df

    def create_quarterly_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Tạo các đặc trưng chênh lệch so với quý trước (QoQ)."""
        df = df.copy()

        # Lấy tất cả các cột cơ bản (ví dụ: NIM, CIR, ROE, v.v.)
        fundamental_cols = [
            'NIM (%)', 'NPL (%)', 'CIR (%)', 'Provision_Coverage (%)',
            'Cost_of_Funds (%)', 'Equity_Ratio (%)', 'Loan_to_Deposit (%)',
            'Credit_Growth (%)', 'Pre_Provision_ROA (%)', 'Post_Tax_ROA (%)',
            'Non_Interest_Income (%)', 'Loan_to_Asset (%)', 'NPL_to_Asset (%)',
            'ROE (%)', 'ROA (%)', 'P/E', 'P/B', 'BVPS (VND)'
            # Thêm các cột khác từ fundamental_data nếu cần
        ]

        feature_cols = []
        for col in fundamental_cols:
            if col in df.columns:
                # Tính toán chênh lệch Quarter-over-Quarter (QoQ)
                df[f'{col}_Diff_QoQ'] = df[col].diff()
                feature_cols.append(col) # Giữ lại giá trị gốc
                feature_cols.append(f'{col}_Diff_QoQ') # Thêm giá trị chênh lệch

        # Thêm các đặc trưng về giá
        df['Close_Pct_Change_QoQ'] = df['Close'].pct_change()
        feature_cols.append('Close') # Giữ 'Close' đã chuẩn hóa
        feature_cols.append('Close_Pct_Change_QoQ')

        # Loại bỏ các cột không phải số ra khỏi danh sách cuối cùng
        final_feature_cols = [col for col in feature_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        return df, final_feature_cols

    def process_all_tickers(self, tickers: List[str] = None):
        if tickers is None:
            tickers = self.config.get('data.tickers', [])

        processed_dir = self.config.get('paths.processed', 'data/processed')
        os.makedirs(processed_dir, exist_ok=True)

        total_files_created = 0

        for ticker in tickers:
            try:
                logger.info(f"Đang xử lý dữ liệu hàng quý cho {ticker}...")

                # 1. Tải dữ liệu cơ bản (đã theo quý)
                fundamental_data = self.db.load_dataframe(f"{ticker}_Fundamental")
                if fundamental_data is None or fundamental_data.empty:
                    logger.warning(f"Bỏ qua {ticker}: Không có dữ liệu cơ bản.")
                    continue

                fundamental_data['time'] = pd.to_datetime(fundamental_data['time'])
                fundamental_data = fundamental_data.sort_values('time').drop_duplicates(subset='time', keep='last')
                # Giữ 'time' là cột thường

                # 2. Tải dữ liệu OHLCV (hàng ngày) và Resample theo Quý
                ohlcv_data = self.db.load_dataframe(f"{ticker}_OHLCV")
                if ohlcv_data is None or ohlcv_data.empty:
                    logger.warning(f"Bỏ qua {ticker}: Không có dữ liệu OHLCV.")
                    continue

                ohlcv_data['time'] = pd.to_datetime(ohlcv_data['time'])
                ohlcv_data = ohlcv_data.sort_values('time')
                # Giữ 'time' là cột thường

                # Resample: Lấy giá 'Close' cuối cùng của mỗi Quý (sử dụng 'QE')
                quarterly_ohlcv = ohlcv_data.resample('QE', on='time')['Close'].last().reset_index()

                # 3. Kết hợp bằng merge_asof
                fundamental_data = fundamental_data.sort_values('time')
                quarterly_ohlcv = quarterly_ohlcv.sort_values('time')

                df_q = pd.merge_asof(
                    quarterly_ohlcv,
                    fundamental_data,
                    on='time',
                    direction='backward' # Tìm bản ghi fundamental gần nhất lùi về quá khứ
                )

                if df_q.empty or df_q['Close'].isnull().all(): # Kiểm tra cả cột 'Close' từ resample
                    logger.warning(f"Bỏ qua {ticker}: Dữ liệu sau khi merge_asof bị rỗng hoặc không có giá Close.")
                    continue

                # Loại bỏ các hàng mà merge_asof không tìm thấy dữ liệu fundamental phù hợp
                # Lấy các cột fundamental thực sự có trong fundamental_data
                actual_fundamental_cols = fundamental_data.columns.difference(['time'])
                df_q = df_q.dropna(subset=actual_fundamental_cols)

                if df_q.empty or len(df_q) < 10: # Cần ít nhất 10 quý để tạo chuỗi và chia train/test
                    logger.warning(f"Bỏ qua {ticker}: Không đủ dữ liệu quý sau khi merge và dropna ({len(df_q)} hàng). Cần ít nhất 10.")
                    continue

                df_q['Ticker'] = ticker

                # 4. Tạo đặc trưng và mục tiêu
                df_q, feature_cols = self.create_quarterly_features(df_q)
                df_q = self.create_quarterly_targets(df_q)

                # 5. Dọn dẹp NaNs cuối cùng (từ diff, shift)
                # Đảm bảo các cột đặc trưng thực sự tồn tại trước khi dropna
                valid_feature_cols_final = [col for col in feature_cols if col in df_q.columns]
                df_q = df_q.dropna(subset=valid_feature_cols_final + ['Target_Direction_t+1Q'])
                if df_q.empty:
                    logger.warning(f"Bỏ qua {ticker}: Dữ liệu rỗng sau khi dropna cuối cùng.")
                    continue

                # 6. Chuẩn hóa (Scaling)
                scaler = StandardScaler()
                # Chỉ chuẩn hóa các cột đặc trưng thực sự tồn tại và hợp lệ
                valid_feature_cols_final = [col for col in valid_feature_cols_final if col in df_q.columns]
                if not valid_feature_cols_final:
                    logger.warning(f"Bỏ qua {ticker}: Không còn đặc trưng hợp lệ sau khi dọn dẹp.")
                    continue
                df_q[valid_feature_cols_final] = scaler.fit_transform(df_q[valid_feature_cols_final])

                # 7. Lưu file RIÊNG LẺ (sử dụng cột 'time' làm index khi lưu)
                df_q = df_q.set_index('time') # Đặt lại index để lưu
                save_path = os.path.join(processed_dir, f"{ticker}_quarterly_features.csv")
                df_q.to_csv(save_path, index=True)

                scaler_path = os.path.join(processed_dir, f"{ticker}_quarterly_scaler.pkl")
                joblib.dump(scaler, scaler_path)

                logger.info(f"Đã lưu file hàng quý cho {ticker} tại {save_path} ({len(df_q)} hàng)")
                total_files_created += 1

            except Exception as e:
                logger.error(f"Lỗi khi xử lý {ticker}: {e}", exc_info=True)

        logger.info(f"Hoàn tất xử lý hàng quý. Đã tạo {total_files_created} file.")
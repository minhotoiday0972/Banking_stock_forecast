# src/features/feature_engineer.py
# (PHIÊN BẢN ĐÃ SỬA LỖI RÒ RỈ DỮ LIỆU)

import pandas as pd
import numpy as np
import os
import joblib
import ta # Thêm import
from datetime import datetime, timedelta
from typing import List, Dict
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_database

logger = get_logger("feature_engineer")

class FeatureEngineer:
    def __init__(self):
        self.config = get_config()
        self.db = get_database()
        self.horizons = self.config.get('models.shared.forecast_horizons', [1, 3, 5, 30, 60, 90])
        # Tăng số lượng features được chọn để cải thiện hiệu suất
        self.top_n_features = self.config.get('features.feature_selection.top_n_features', 30)

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán "Vũ trụ Đặc trưng" (Feature Universe), bao gồm cả 
        các chỉ báo ngắn hạn và dài hạn, CÙNG VỚI các đặc trưng bắt trend.
        (Hàm này đã OK, giữ nguyên)
        """
        df = df.copy()
        
        # === CÁC ĐẶC TRƯNG NGẮN HẠN (Daily) ===
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()
        df['Close_MA14'] = df['Close'].rolling(window=14).mean()
        df['Close_MA30'] = df['Close'].rolling(window=30).mean()
        df['Close_to_Open'] = (df['Close'] - df['Open']) / df['Open']
        df['High_to_Low'] = (df['High'] - df['Low']) / df['Low']
        df['Volatility_14'] = df['Close'].rolling(window=14).std()
        df['Close_Pct_Change'] = df['Close'].pct_change()
        
        # RSI 14
        delta_14 = df['Close'].diff()
        gain_14 = (delta_14.where(delta_14 > 0, 0)).rolling(window=14).mean()
        loss_14 = (-delta_14.where(delta_14 < 0, 0)).rolling(window=14).mean()
        rs_14 = gain_14 / loss_14
        df['RSI_14'] = 100 - (100 / (1 + rs_14))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands (20)
        rolling_mean_20 = df['Close'].rolling(window=20).mean()
        rolling_std_20 = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = rolling_mean_20 + (rolling_std_20 * 2)
        df['BB_Lower'] = rolling_mean_20 - (rolling_std_20 * 2)
        
        # === BỔ SUNG ĐẶC TRƯNG DÀI HẠN (Quarterly/Yearly) ===
        logger.debug("Đang tính toán các đặc trưng kỹ thuật dài hạn...")
        df['Close_MA100'] = df['Close'].rolling(window=100).mean()
        df['Close_MA200'] = df['Close'].rolling(window=200).mean()
        df['Volatility_60'] = df['Close'].rolling(window=60).std() # Biến động 3 tháng
        
        # RSI 30 (dài hạn hơn)
        delta_30 = df['Close'].diff()
        gain_30 = (delta_30.where(delta_30 > 0, 0)).rolling(window=30).mean()
        loss_30 = (-delta_30.where(delta_30 < 0, 0)).rolling(window=30).mean()
        rs_30 = gain_30 / loss_30
        df['RSI_30'] = 100 - (100 / (1 + rs_30))
        
        # ====================================================================
        # === BỔ SUNG MỚI: ĐẶC TRƯNG THỜI GIAN (ĐỂ BẮT TREND/LẠM PHÁT) ===
        # ====================================================================
        logger.debug("Đang tính toán các đặc trưng thời gian (trend features)...")
        
        # 1. Chỉ số thời gian tuyến tính (Linear Time Index)
        # Giả định df đã được sort_values('time') và reset_index() trong process_all_tickers
        df['time_index'] = df.index
        
        # 2. Các đặc trưng Lịch (Calendar Features)
        # Cột 'time' là một cột datetime (đã được convert trong process_all_tickers)
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['day_of_week'] = df['time'].dt.dayofweek
        
        # Cột 'year' có thể đã có từ Fundamental, nhưng ta thêm ở đây
        # để đảm bảo nó luôn tồn tại, ngay cả khi thiếu dữ liệu Fundamental.
        if 'year' not in df.columns:
            logger.warning("Không có cột 'year' từ fundamental, đang tự tạo từ 'time'.")
            df['year'] = df['time'].dt.year

        # ====================================================================
        # === BỔ SUNG MỚI: ĐẶC TRƯNG LAG DÀI HẠN (ĐỂ BẮT BỐI CẢNH) ===
        # ====================================================================
        logger.debug("Đang tính toán các đặc trưng lag dài hạn...")
        df['Close_lag_90'] = df['Close'].shift(90)   # Giá của 90 ngày giao dịch trước
        df['Close_lag_365'] = df['Close'].shift(365) # Giá của 365 ngày giao dịch trước
        # === KẾT THÚC BỔ SUNG MỚI ===
        
        return df

    def calculate_banking_features(self, df: pd.DataFrame, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        key_banking_ratios = ['NPL (%)', 'NIM (%)', 'CIR (%)', 'Credit_Growth (%)']
        for ratio in key_banking_ratios:
            if ratio in df.columns:
                df[ratio.replace(' (%)', '_Diff')] = df[ratio].diff()
                df[ratio.replace(' (%)', '_MA4')] = df[ratio].rolling(window=4).mean()
        
        if 'NIM (%)' in df.columns and 'CIR (%)' in df.columns:
            df['NIM_CIR_Ratio'] = df['NIM (%)'] / df['CIR (%)']
            
        if fundamental_data is not None and 'NPL (%)' in fundamental_data.columns:
            # Đảm bảo 'time' là index cho resample
            try:
                fundamental_data = fundamental_data.set_index('time')
            except KeyError:
                pass # 'time' đã là index
            quarterly_npl = fundamental_data['NPL (%)'].dropna().resample('QE').last().to_frame()
            quarterly_npl['NPL_Trend'] = quarterly_npl['NPL (%)'].diff().apply(lambda x: 1 if x < 0 else 0)
            
            # Đưa 'time' trở lại làm cột để merge_asof
            fundamental_data = fundamental_data.reset_index()
            
            # === SỬA LỖI RÒ RỈ 1 ===
            # Đổi direction='forward' (nhìn về tương lai) 
            # thành 'backward' (nhìn về quá khứ).
            df = pd.merge_asof(df.sort_values('time'), 
                              quarterly_npl[['NPL_Trend']].reset_index(), 
                              on='time', 
                              direction='backward') # <- ĐÃ SỬA
            # === KẾT THÚC SỬA ===
            
        return df

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        # Sửa đổi: Chỉ hỗ trợ 2 lớp (Down, Up) dựa trên biến động giá.
        # Class 0: Down (Giá giảm hoặc không đổi)
        # Class 1: Up (Giá tăng)
        """
        df = df.copy()
        
        # Không cần dùng ngưỡng động nữa cho việc tạo nhãn 2 lớp
        # logger.info("Đang tạo nhãn 2 lớp (Giảm/Tăng) với ngưỡng động theo horizon (loại bỏ lớp Neutral)...")
        
        logger.info("Đang tạo nhãn 2 lớp (Giảm/Tăng) dựa trên biến động giá (ngưỡng 0%)...")

        for horizon in self.horizons:
            future_price = df['Close'].shift(-horizon)
            price_change = (future_price - df['Close']) / df['Close']
            
            # Điều kiện cho 2 lớp (Up=1 nếu >0, Down=0 nếu <=0)
            conditions = [
                price_change > 0  # Lớp 1 (Up)
            ]
            choices = [1]
            
            # Gán nhãn. Nếu không thỏa mãn conditions (tức là price_change <= 0), gán default=0 (Down)
            df[f'Target_Direction_t+{horizon}'] = np.select(conditions, choices, default=0)
            
            # Không còn các mẫu Neutral để loại bỏ
            
            # In phân bổ lớp để kiểm tra (chỉ các hàng không phải NaN)
            class_distribution = df[f'Target_Direction_t+{horizon}'].dropna().value_counts().to_dict()
            logger.debug(f"    Phân bổ lớp cho t+{horizon}: {class_distribution}")

        return df

    def _clean_data(self, df: pd.DataFrame, all_feature_cols: List[str]) -> pd.DataFrame:
        """Chỉ fillna cho các cột đặc trưng, giữ lại NaN ở mục tiêu. (Giữ nguyên)"""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        
        for col in all_feature_cols:
            if col not in df.columns: continue # Bỏ qua nếu cột không tồn tại
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().all():
                    df[col] = 0
                else:
                    df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].ffill().bfill()
        return df

    def _get_golden_features(self, df: pd.DataFrame, candidate_cols: List[str], target_col: str) -> List[str]:
        """Chạy RF để tìm các đặc trưng 'vàng'. (Giữ nguyên)"""
        logger.info(f"Đang chạy lựa chọn đặc trưng cho {target_col} từ {len(candidate_cols)} ứng cử viên...")
        
        # Đảm bảo candidate_cols chỉ chứa các cột thực sự có trong df
        valid_candidate_cols = [col for col in candidate_cols if col in df.columns]
        
        df_clean = df.dropna(subset=[target_col] + valid_candidate_cols)
        
        if df_clean.empty or len(df_clean) < 50: # Yêu cầu ít nhất 50 mẫu
            logger.warning(f"Không đủ dữ liệu ({len(df_clean)} mẫu) để lựa chọn đặc trưng cho {target_col}, sử dụng tất cả ứng cử viên.")
            return valid_candidate_cols
            
        X = df_clean[valid_candidate_cols]
        y = df_clean[target_col].astype(int)
        
        # Lọc tương quan cao
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
        X = X.drop(columns=to_drop_corr, errors='ignore')
        
        # Lọc đặc trưng không đổi
        variances = X.var()
        constant_columns = variances[variances == 0].index.tolist()
        X = X.drop(columns=constant_columns, errors='ignore')
        
        if X.empty:
             logger.warning(f"Không còn đặc trưng nào sau khi lọc cho {target_col}, sử dụng tất cả ứng cử viên.")
             return valid_candidate_cols

        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
        model.fit(X, y)
        
        importances = model.feature_importances_
        results = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
        
        # Lấy Top N đặc trưng
        golden_features = [feature for feature, importance in results[:self.top_n_features]]
        logger.info(f"Đã chọn {len(golden_features)} đặc trưng 'vàng' cho {target_col} (Top 3: {golden_features[:3]}).")
        
        return golden_features

    def process_all_tickers(self, tickers: List[str] = None):
        """
        HÀM QUAN TRỌNG NHẤT
        Đã sửa 2 lỗi rò rỉ dữ liệu (merge và scaler).
        """
        if tickers is None:
            tickers = self.config.get('data.tickers', [])
        start_date = self.config.get('data.start_date')
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        processed_dir = self.config.get('paths.processed', 'data/processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # 1. Tải Dữ liệu Thị trường (VNINDEX)
        vnindex_path = os.path.join(self.config.get('data.raw_dir', 'data/raw'), "VNINDEX.csv")
        market_df = None
        if os.path.exists(vnindex_path):
            market_df = pd.read_csv(vnindex_path, parse_dates=['time'])
            market_df['Market_Pct_Change'] = market_df['VNINDEX'].pct_change()
            market_df['Market_Volatility'] = market_df['VNINDEX'].rolling(window=14).std()
            market_df = market_df[['time', 'Market_Pct_Change', 'Market_Volatility']]
        else:
            logger.error("Không tìm thấy file VNINDEX.csv. Bỏ qua đặc trưng thị trường.")
            
        results = {}
        for ticker in tickers:
            try:
                # 2. Tải và hợp nhất dữ liệu (OHLCV + Fundamental)
                ohlcv_data = self.db.load_dataframe(f"{ticker}_OHLCV")
                if ohlcv_data is None:
                    logger.warning(f"Không có dữ liệu OHLCV cho {ticker}, bỏ qua."); continue
                ohlcv_data['time'] = pd.to_datetime(ohlcv_data['time'])
                ohlcv_data = ohlcv_data[(ohlcv_data['time'] >= start_date) & (ohlcv_data['time'] <= end_date)].sort_values('time').reset_index(drop=True)
                
                fundamental_data = self.db.load_dataframe(f"{ticker}_Fundamental")
                if fundamental_data is not None and not fundamental_data.empty:
                    fundamental_data['time'] = pd.to_datetime(fundamental_data['time'])
                    fundamental_data = fundamental_data.sort_values('time').drop_duplicates(subset='time', keep='last')
                    
                    # === SỬA LỖI RÒ RỈ 2 ===
                    # Đổi direction='forward' (nhìn về tương lai) 
                    # thành 'backward' (nhìn về quá khứ).
                    logger.info(f"Đang merge_asof dữ liệu cơ bản cho {ticker} (direction='backward')...")
                    df = pd.merge_asof(ohlcv_data, fundamental_data, 
                                      on='time', 
                                      direction='backward') # <- ĐÃ SỬA
                    # === KẾT THÚC SỬA ===
                else:
                    df = ohlcv_data
                    logger.warning(f"No fundamental data found for {ticker}")
                
                # Hợp nhất Dữ liệu Thị trường
                if market_df is not None:
                    df = pd.merge(df, market_df, on='time', how='left')
                    
                df['Ticker'] = ticker

                # 3. Tính toán "Vũ trụ Đặc trưng" (Feature Universe)
                df = self.calculate_technical_indicators(df)
                if fundamental_data is not None and not fundamental_data.empty:
                    df = self.calculate_banking_features(df, fundamental_data)
                
                # 4. Tạo TẤT CẢ các mục tiêu
                df = self.create_targets(df)
                
                # 5. Xác định các "Vũ trụ con" (Sub-Universes)
                all_available_features = [
                    col for col in df.columns 
                    if not col.startswith('Target_') and col not in ['Ticker', 'time', 'Open', 'High', 'Low'] 
                    and pd.api.types.is_numeric_dtype(df[col])
                ]
                
                # --- Định nghĩa các vũ trụ đặc trưng ---
                SHORT_TERM_TECHNICAL = [
                    'Close_MA7', 'Close_MA14', 'Close_MA30', 'Close_to_Open', 'High_to_Low',
                    'Volatility_14', 'Close_Pct_Change', 'RSI_14', 'MACD', 'MACD_Signal',
                    'BB_Upper', 'BB_Lower', 'Market_Pct_Change', 'Market_Volatility', 'Volume'
                ]
                
                FUNDAMENTAL_DIFFS = [
                    'NPL_Diff', 'NIM_Diff', 'CIR_Diff', 'Credit_Growth_Diff', 'NPL_Trend'
                ]
                
                SHORT_TERM_UNIVERSE = [
                    col for col in all_available_features 
                    if col in SHORT_TERM_TECHNICAL or col in FUNDAMENTAL_DIFFS
                ]
                
                LONG_TERM_UNIVERSE = all_available_features
                # --- KẾT THÚC ĐỊNH NGHĨA ---
                
                # 6. Dọn dẹp (Chỉ Fillna)
                if 'Open' in df.columns:
                    df.drop(columns=['Open', 'High', 'Low'], inplace=True)
                
                all_available_features = [col for col in all_available_features if col in df.columns]
                
                df = self._clean_data(df, all_available_features)
                
                # === SỬA LỖI RÒ RỈ 3 (SCALER) ===
                # Chúng ta sẽ fit scaler CHỈ trên 80% dữ liệu đầu
                # (để mô phỏng tập train) và transform cho toàn bộ.
                
                # Tính toán tập "train proxy" để fit scaler
                # Lấy 80% dữ liệu đầu (sau khi đã clean data)
                train_proxy_size = int(len(df) * 0.8)
                df_train_proxy = df.iloc[:train_proxy_size]
                
                if df_train_proxy.empty or len(df_train_proxy) < 50:
                    logger.error(f"Không đủ dữ liệu (Proxy={len(df_train_proxy)}) để fit scaler cho {ticker}, bỏ qua.")
                    continue
                logger.info(f"Đang fit scaler dựa trên {len(df_train_proxy)} mẫu (80% dữ liệu đầu).")
                # -----------------------------------

                # 7. Lặp qua từng Horizon để tạo Metadata riêng biệt
                all_scalers = {} 
                
                for h in self.horizons:
                    target_col = f'Target_Direction_t+{h}'
                    
                    # --- Logic "Bộ lọc Thông minh" ---
                    if h in [1, 3, 5]:
                        logger.info(f"Sử dụng 'Vũ trụ Đặc trưng Ngắn hạn' cho t+{h}...")
                        candidate_cols = SHORT_TERM_UNIVERSE
                    else: # 30, 60, 90
                        logger.info(f"Sử dụng 'Vũ trụ Đặc trưng Đầy đủ' cho t+{h}...")
                        candidate_cols = LONG_TERM_UNIVERSE
                    # --- KẾT THÚC ---
                    
                    # 7a. Chạy lựa chọn đặc trưng tự động
                    golden_features_h = self._get_golden_features(df, candidate_cols, target_col)
                    
                    if not golden_features_h:
                         logger.warning(f"Không tìm thấy golden features cho {ticker} t+{h}. Bỏ qua horizon này.")
                         continue
                    
                    # 7b. Tạo và Fit Scaler (ĐÃ SỬA)
                    scaler = MinMaxScaler()
                    # Fit CHỈ trên 80% dữ liệu (train_proxy)
                    scaler.fit(df_train_proxy[golden_features_h]) # <- ĐÃ SỬA
                    all_scalers[h] = scaler # Lưu scaler cho horizon này
                    
                    # 7c. Lưu Metadata riêng
                    metadata = {
                        'feature_columns': golden_features_h,
                        'scaler': scaler
                    }
                    metadata_path = os.path.join(processed_dir, f"{ticker}_metadata_t+{h}.pkl")
                    joblib.dump(metadata, metadata_path)
                
                # 8. Chuẩn hóa và Lưu 1 file dữ liệu lớn (ĐÃ SỬA)
                main_scaler = MinMaxScaler()
                # Fit CHỈ trên 80% dữ liệu (train_proxy)
                logger.info(f"Fit Main Scaler trên {len(df_train_proxy)} mẫu (proxy)...")
                main_scaler.fit(df_train_proxy[all_available_features]) # <- ĐÃ SỬA
                
                # Transform trên TOÀN BỘ 100% dữ liệu
                logger.info(f"Transform Main Scaler trên {len(df)} mẫu (toàn bộ)...")
                df[all_available_features] = main_scaler.transform(df[all_available_features]) # <- ĐÃ SỬA
                
                features_path = os.path.join(processed_dir, f"{ticker}_features_scaled.csv")
                df.to_csv(features_path, index=False)
                
                main_scaler_path = os.path.join(processed_dir, f"{ticker}_main_scaler.pkl")
                joblib.dump(main_scaler, main_scaler_path)
                # === KẾT THÚC SỬA LỖI 3 ===

                logger.info(f"Đã xử lý và lưu dữ liệu/metadata (an toàn) cho {ticker}")
                results[ticker] = True
            except Exception as e:
                logger.error(f"Lỗi khi xử lý {ticker}: {e}", exc_info=True)
                results[ticker] = False
        return results
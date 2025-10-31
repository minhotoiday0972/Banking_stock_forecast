# src/features/feature_engineer.py
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from typing import List
from sklearn.preprocessing import MinMaxScaler

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_database

logger = get_logger("feature_engineer")

class FeatureEngineer:
    def __init__(self):
        self.config = get_config()
        self.db = get_database()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán TẤT CẢ các chỉ báo kỹ thuật.
        Chúng ta vẫn tính tất cả, nhưng sẽ lọc ra sau.
        """
        df = df.copy()
        
        # Các đặc trưng config yêu cầu
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()
        df['Close_MA30'] = df['Close'].rolling(window=30).mean()
        df['Close_to_Open'] = (df['Close'] - df['Open']) / df['Open']
        df['High_to_Low'] = (df['High'] - df['Low']) / df['Low']
        
        # Các đặc trưng đã có
        df['Close_MA14'] = df['Close'].rolling(window=14).mean()
        df['Volatility_14'] = df['Close'].rolling(window=14).std()
        df['Close_Pct_Change'] = df['Close'].pct_change()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df['BB_Lower'] = rolling_mean - (rolling_std * 2)
        
        return df

    def calculate_banking_features(self, df: pd.DataFrame, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán TẤT CẢ các chỉ báo cơ bản.
        Chúng ta vẫn tính tất cả, nhưng sẽ lọc ra sau.
        """
        df = df.copy()
        key_banking_ratios = ['NPL (%)', 'NIM (%)', 'CIR (%)', 'Credit_Growth (%)']
        for ratio in key_banking_ratios:
            if ratio in df.columns:
                df[ratio.replace(' (%)', '_Diff')] = df[ratio].diff()
                df[ratio.replace(' (%)', '_MA4')] = df[ratio].rolling(window=4).mean()
        
        if 'NIM (%)' in df.columns and 'CIR (%)' in df.columns:
            df['NIM_CIR_Ratio'] = df['NIM (%)'] / df['CIR (%)']
            
        if fundamental_data is not None and 'NPL (%)' in fundamental_data.columns:
            quarterly_npl = fundamental_data[['time', 'NPL (%)']].dropna().set_index('time').resample('QE').last()
            quarterly_npl['NPL_Trend'] = quarterly_npl['NPL (%)'].diff().apply(lambda x: 1 if x < 0 else 0)
            df = pd.merge_asof(df.sort_values('time'), quarterly_npl[['NPL_Trend']].reset_index(), on='time', direction='forward')
            
        return df

    def create_targets(self, df: pd.DataFrame, horizons: List[int] = None) -> pd.DataFrame:
        if horizons is None:
            horizons = self.config.get('models.shared.forecast_horizons', [1, 3, 5])
        df = df.copy()
        for horizon in horizons:
            df[f'Target_Close_t+{horizon}'] = df['Close'].shift(-horizon)
            future_price = df['Close'].shift(-horizon)
            price_change = (future_price - df['Close']) / df['Close']
            df[f'Target_Direction_t+{horizon}'] = np.where(price_change > 0, 1, 0)
        return df

    def _remove_highly_correlated_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Loại bỏ các đặc trưng tương quan cao CHỈ TỪ DANH SÁCH ĐÃ CHỌN."""
        if not feature_cols:
            return df
        
        corr_matrix = df[feature_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Chỉ tìm trong các cột đặc trưng đã chọn
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95) and column in feature_cols]
        
        if to_drop:
            # Không drop khỏi df, chỉ loại khỏi danh sách
            feature_cols = [col for col in feature_cols if col not in to_drop]
            logger.info(f"Đã loại bỏ đặc trưng tương quan cao: {to_drop}")
            
        return df, feature_cols

    def _remove_constant_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Loại bỏ các đặc trưng không đổi CHỈ TỪ DANH SÁCH ĐÃ CHỌN."""
        if not feature_cols:
            return df
        
        variances = df[feature_cols].var()
        constant_columns = variances[variances == 0].index.tolist()
        
        if constant_columns:
            feature_cols = [col for col in feature_cols if col not in constant_columns]
            logger.info(f"Đã loại bỏ đặc trưng không đổi: {constant_columns}")
            
        return df, feature_cols

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chỉ fillna, không dropna mục tiêu."""
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)

        feature_cols = [c for c in df.columns if not c.startswith('Target_') and c not in ['Ticker', 'time']]
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().all():
                    df[col] = 0
                else:
                    col_median = df[col].median()
                    df[col] = df[col].fillna(col_median)
            else:
                df[col] = df[col].ffill().bfill()
        
        return df

    def process_all_tickers(self, tickers: List[str] = None):
        if tickers is None:
            tickers = self.config.get('data.tickers', [])
        start_date = self.config.get('data.start_date')
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 1. Tải Dữ liệu Thị trường (VNINDEX)
        vnindex_path = os.path.join(self.config.get('paths.raw', 'data/raw'), "VNINDEX.csv")
        if not os.path.exists(vnindex_path):
            logger.error("Không tìm thấy file VNINDEX.csv. Bỏ qua đặc trưng thị trường.")
            market_df = None
        else:
            market_df = pd.read_csv(vnindex_path, parse_dates=['time'])
            market_df['Market_Pct_Change'] = market_df['VNINDEX'].pct_change()
            market_df['Market_Volatility'] = market_df['VNINDEX'].rolling(window=14).std()
            market_df = market_df[['time', 'Market_Pct_Change', 'Market_Volatility']]
            
        results = {}
        for ticker in tickers:
            try:
                # 2. Tải và hợp nhất dữ liệu (OHLCV + Fundamental)
                ohlcv_data = self.db.load_dataframe(f"{ticker}_OHLCV")
                ohlcv_data['time'] = pd.to_datetime(ohlcv_data['time'])
                ohlcv_data = ohlcv_data[(ohlcv_data['time'] >= start_date) & (ohlcv_data['time'] <= end_date)].sort_values('time').reset_index(drop=True)
                
                fundamental_data = self.db.load_dataframe(f"{ticker}_Fundamental")
                if fundamental_data is not None and not fundamental_data.empty:
                    fundamental_data['time'] = pd.to_datetime(fundamental_data['time'])
                    fundamental_data = fundamental_data.sort_values('time').drop_duplicates(subset='time', keep='last')
                    df = pd.merge_asof(ohlcv_data, fundamental_data, on='time', direction='forward')
                else:
                    df = ohlcv_data
                    logger.warning(f"No fundamental data found for {ticker}")
                
                # Hợp nhất Dữ liệu Thị trường
                if market_df is not None:
                    df = pd.merge(df, market_df, on='time', how='left')
                    
                df['Ticker'] = ticker

                # 3. Tính toán TẤT CẢ đặc trưng
                df = self.calculate_technical_indicators(df)
                if fundamental_data is not None and not fundamental_data.empty:
                    df = self.calculate_banking_features(df, fundamental_data)
                df = self.create_targets(df)
                
                # 4. Dọn dẹp dữ liệu (Fillna)
                # Bỏ 'Open', 'High', 'Low' thô sau khi đã dùng để tính toán
                if 'Open' in df.columns:
                    df.drop(columns=['Open', 'High', 'Low'], inplace=True)
                df = self._clean_data(df) # Hàm này chỉ fillna
                
                # --- THAY ĐỔI QUAN TRỌNG: Lọc Đặc Trưng (Pruning) ---
                
                # Danh sách "vàng" (Golden List) dựa trên kết quả evaluate_features.py
                # Bao gồm cả các đặc trưng cơ bản và thị trường
                GOLDEN_FEATURES = [
                    # Nhóm Kỹ thuật (Tín hiệu mạnh nhất)
                    'MACD_Signal', 
                    'MACD', 
                    'Volatility_14', 
                    'RSI_14', 
                    'Volume', 
                    'Close_Pct_Change',
                    'Close_MA7',
                    'Close_MA30',
                    'Close_to_Open',
                    'High_to_Low',
                    'BB_Upper', # Bollinger Bands
                    'BB_Lower',
                    
                    # Nhóm Thị trường (Bổ sung bối cảnh)
                    'Market_Pct_Change',
                    'Market_Volatility',
                    
                    # Nhóm Cơ bản (Tín hiệu yếu nhưng vẫn giữ lại vài cái tốt nhất)
                    'Cost_of_Funds (%)',
                    'CIR (%)',
                    'P/E',
                    'NIM_CIR_Ratio',
                    'Post_Tax_ROA (%)',
                    'P/B',
                    'Non_Interest_Income (%)',
                    'Credit_Growth (%)',
                    'NIM_Diff', # Giữ lại các tín hiệu 'Diff'
                    'CIR_Diff',
                    'NPL_Diff'
                ]
                
                # Lấy tất cả các cột đặc trưng có sẵn
                all_available_features = [
                    col for col in df.columns 
                    if not col.startswith('Target_') and col not in ['Ticker', 'time'] 
                    and pd.api.types.is_numeric_dtype(df[col])
                ]
                
                # Lọc lần 1: Chỉ chọn các đặc trưng có trong GOLDEN_FEATURES
                feature_cols = [col for col in all_available_features if col in GOLDEN_FEATURES]
                
                logger.info(f"Đã lọc (bước 1): Giữ lại {len(feature_cols)}/{len(all_available_features)} đặc trưng 'vàng'.")

                # Lọc lần 2: Loại bỏ tương quan và không đổi
                df, feature_cols = self._remove_highly_correlated_features(df, feature_cols)
                df, feature_cols = self._remove_constant_features(df, feature_cols)
                
                logger.info(f"Đã lọc (bước 2): {len(feature_cols)} đặc trưng cuối cùng được chọn.")
                # --- KẾT THÚC THAY ĐỔI ---
                
                
                # 5. Scaling và lưu metadata (CHỈ với các đặc trưng đã lọc)
                target_cols_reg = [col for col in df.columns if col.startswith('Target_Close')]

                metadata = {
                    'feature_columns': feature_cols, # Chỉ lưu các đặc trưng cuối cùng
                    'scalers': {}
                }
                
                if feature_cols:
                    feature_scaler = MinMaxScaler()
                    df_features_scaled = feature_scaler.fit_transform(df[feature_cols])
                    df[feature_cols] = df_features_scaled
                    metadata['scalers']['features'] = feature_scaler
                
                for col in target_cols_reg:
                    target_scaler = MinMaxScaler()
                    valid_data = df[[col]].dropna()
                    if not valid_data.empty:
                        scaled_data = target_scaler.fit_transform(valid_data)
                        df.loc[valid_data.index, col] = scaled_data
                        metadata['scalers'][col] = target_scaler

                processed_dir = self.config.get('paths.processed', 'data/processed')
                os.makedirs(processed_dir, exist_ok=True)
                joblib.dump(metadata, os.path.join(processed_dir, f"{ticker}_metadata.pkl"))

                # 6. Lưu file đã scale
                features_path = os.path.join(processed_dir, f"{ticker}_features_scaled.csv")
                df.to_csv(features_path, index=False)
                
                logger.info(f"Successfully processed and saved metadata and scaled features for {ticker}")
                results[ticker] = True
            except Exception as e:
                logger.error(f"Failed to process features for {ticker}: {e}", exc_info=True)
                results[ticker] = False
        return results
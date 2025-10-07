# src/features/feature_engineer.py
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_database

logger = get_logger("feature_engineer")

class FeatureEngineer:
    """Centralized feature engineering"""
    
    def __init__(self):
        self.config = get_config()
        self.db = get_database()
        self.feature_columns = self._get_feature_columns()
    
    def _get_feature_columns(self) -> List[str]:
        """Get feature columns from config"""
        technical = self.config.get('features.technical', [])
        fundamental = self.config.get('features.fundamental', [])
        banking_specific = self.config.get('features.banking_specific', [])
        market = self.config.get('features.market', [])
        
        # Add derived features
        derived = [
            'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Momentum_20D'
        ]
        
        return technical + fundamental + banking_specific + market + derived
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # Moving averages
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()
        df['Close_MA14'] = df['Close'].rolling(window=14).mean()
        df['Close_MA30'] = df['Close'].rolling(window=30).mean()
        
        # Volatility
        df['Volatility_14'] = df['Close'].rolling(window=14).std()
        
        # Price ratios
        df['Close_to_Open'] = df['Close'] / df['Open']
        df['High_to_Low'] = df['High'] / df['Low']
        
        # Returns
        df['Close_Pct_Change'] = df['Close'].pct_change()
        
        # RSI
        df['RSI_14'] = self._calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
        
        # Momentum
        df['Momentum_20D'] = df['Close'].diff(20)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(0)
    
    def _calculate_macd(self, prices: pd.Series, short: int = 12, 
                       long: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line"""
        exp1 = prices.ewm(span=short, adjust=False).mean()
        exp2 = prices.ewm(span=long, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal, adjust=False).mean()
        return macd.fillna(0), signal.fillna(0)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, 
                                  num_std: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def calculate_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate fundamental features"""
        df = df.copy()
        
        # Fundamental differences
        if 'ROE (%)' in df.columns:
            df['ROE_Diff'] = df['ROE (%)'].diff()
        
        if 'P/E' in df.columns:
            df['P_E_Diff'] = df['P/E'].diff()
            df['PE_to_Close'] = df['P/E'] / df['Close']
        
        return df
    
    def calculate_banking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate banking-specific features"""
        df = df.copy()
        
        # Banking ratio differences (trend analysis)
        key_banking_ratios = ['NPL (%)', 'NIM (%)', 'CIR (%)', 'Credit_Growth (%)']
        
        for ratio in key_banking_ratios:
            if ratio in df.columns:
                # Calculate difference (trend)
                diff_col = ratio.replace(' (%)', '_Diff')
                df[diff_col] = df[ratio].diff()
                
                # Calculate moving averages for smoothing
                ma_col = ratio.replace(' (%)', '_MA4')
                df[ma_col] = df[ratio].rolling(window=4).mean()
        
        # Banking efficiency ratios
        if 'NIM (%)' in df.columns and 'CIR (%)' in df.columns:
            # Efficiency ratio: NIM/CIR (higher is better)
            df['NIM_CIR_Ratio'] = df['NIM (%)'] / df['CIR (%)']
        
        if 'NPL (%)' in df.columns and 'Provision_Coverage (%)' in df.columns:
            # Risk coverage ratio (how well provisions cover bad loans)
            df['Risk_Coverage_Ratio'] = df['Provision_Coverage (%)'] / df['NPL (%)']
        
        # Asset quality indicators
        if 'NPL (%)' in df.columns:
            # NPL trend (1 if improving, 0 if worsening)
            df['NPL_Trend'] = df['NPL (%)'].rolling(window=4).apply(
                lambda x: 1 if x.iloc[-1] < x.iloc[0] else 0, raw=False
            )
        
        # Profitability efficiency
        if 'Pre_Provision_ROA (%)' in df.columns and 'Post_Tax_ROA (%)' in df.columns:
            # Credit loss impact on profitability
            df['Credit_Loss_Impact'] = df['Pre_Provision_ROA (%)'] - df['Post_Tax_ROA (%)']
        
        # Funding efficiency
        if 'Cost_of_Funds (%)' in df.columns and 'NIM (%)' in df.columns:
            # Net spread (NIM - Cost of Funds)
            df['Net_Spread'] = df['NIM (%)'] - df['Cost_of_Funds (%)']
        
        # Asset utilization
        if 'Loan_to_Asset (%)' in df.columns and 'Credit_Growth (%)' in df.columns:
            # Asset deployment efficiency
            df['Asset_Deployment'] = df['Loan_to_Asset (%)'] * (1 + df['Credit_Growth (%)'] / 100)
        
        # Balance sheet strength
        if 'Equity_Ratio (%)' in df.columns and 'NPL (%)' in df.columns:
            # Capital buffer relative to risk
            df['Capital_Risk_Buffer'] = df['Equity_Ratio (%)'] / df['NPL (%)']
        
        # Loan portfolio quality
        if 'Loan_to_Deposit (%)' in df.columns:
            # Loan-to-deposit sustainability (should be < 100%)
            df['LDR_Sustainability'] = 100 - df['Loan_to_Deposit (%)']
        
        return df
    
    def create_targets(self, df: pd.DataFrame, horizons: List[int] = None) -> pd.DataFrame:
        """Create target variables for different horizons"""
        if horizons is None:
            horizons = self.config.get('models.forecast_horizons', [1, 3, 5])
        
        df = df.copy()
        
        for horizon in horizons:
            # Regression target (future close price)
            df[f'Target_Close_t+{horizon}'] = df['Close'].shift(-horizon)
            
            # Classification target (price direction)
            future_price = df['Close'].shift(-horizon)
            current_price = df['Close']
            
            # Create 3-class target: 0=Down, 1=Flat, 2=Up
            price_change = (future_price - current_price) / current_price
            conditions = [
                price_change < -0.01,  # Down > 1% (more sensitive)
                (price_change >= -0.01) & (price_change <= 0.01),  # Flat ±1% (narrower)
                price_change > 0.01   # Up > 1% (more sensitive)
            ]
            choices = [0, 1, 2]
            df[f'Target_Direction_t+{horizon}'] = np.select(conditions, choices, default=1)
        
        return df
    
    def load_market_features(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load market-wide features"""
        try:
            all_ohlcv = self.db.load_dataframe('All_OHLCV')
            if all_ohlcv is None:
                logger.warning("No All_OHLCV data found")
                return {}
            
            all_ohlcv['time'] = pd.to_datetime(all_ohlcv['time'])
            all_ohlcv = all_ohlcv[
                (all_ohlcv['time'] >= start_date) & 
                (all_ohlcv['time'] <= end_date)
            ].sort_values('time')
            
            # Calculate market features
            market_avg = all_ohlcv.groupby('time')['Close'].mean().reset_index()
            market_avg.columns = ['time', 'Market_Avg_Close']
            
            market_vol = all_ohlcv.groupby('time')['Close'].std().reset_index()
            market_vol.columns = ['time', 'Market_Volatility']
            
            market_volume = all_ohlcv.groupby('time')['Volume'].mean().reset_index()
            market_volume.columns = ['time', 'Market_Avg_Volume']
            
            # Fill NaN values
            for df in [market_avg, market_vol, market_volume]:
                for col in df.columns:
                    if col != 'time':
                        df[col] = df[col].fillna(df[col].median())
            
            return {
                'market_avg': market_avg,
                'market_vol': market_vol,
                'market_volume': market_volume
            }
            
        except Exception as e:
            logger.error(f"Error loading market features: {e}")
            return {}
    
    def process_ticker(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Process features for a single ticker"""
        try:
            logger.info(f"Processing features for {ticker}")
            
            # Load OHLCV data
            ohlcv_data = self.db.load_dataframe(f"{ticker}_OHLCV")
            if ohlcv_data is None:
                raise ValueError(f"No OHLCV data for {ticker}")
            
            ohlcv_data['time'] = pd.to_datetime(ohlcv_data['time'])
            ohlcv_data = ohlcv_data[
                (ohlcv_data['time'] >= start_date) & 
                (ohlcv_data['time'] <= end_date)
            ].sort_values('time').reset_index(drop=True)
            
            if ohlcv_data.empty:
                raise ValueError(f"No OHLCV data for {ticker} in date range")
            
            # Load fundamental data
            fundamental_data = self.db.load_dataframe(f"{ticker}_Fundamental")
            if fundamental_data is not None:
                fundamental_data['time'] = pd.to_datetime(fundamental_data['time'])
                fundamental_data = fundamental_data.sort_values('time')
                
                # Define fundamental columns to merge (including banking specific)
                fundamental_cols = ['time', 'ROE (%)', 'ROA (%)', 'P/E', 'P/B', 'BVPS (VND)']
                banking_cols = [
                    'NIM (%)', 'NPL (%)', 'CIR (%)', 'Provision_Coverage (%)',
                    'Cost_of_Funds (%)', 'Equity_Ratio (%)', 'Loan_to_Deposit (%)',
                    'Credit_Growth (%)', 'Pre_Provision_ROA (%)', 'Post_Tax_ROA (%)',
                    'Non_Interest_Income (%)', 'Loan_to_Asset (%)', 'NPL_to_Asset (%)'
                ]
                
                # Add available banking columns
                available_banking_cols = [col for col in banking_cols if col in fundamental_data.columns]
                merge_cols = fundamental_cols + available_banking_cols
                
                # Merge with OHLCV using backward fill
                df = pd.merge_asof(
                    ohlcv_data.sort_values('time'),
                    fundamental_data[merge_cols],
                    on='time',
                    direction='backward'
                )
            else:
                df = ohlcv_data.copy()
                logger.warning(f"No fundamental data for {ticker}")
            
            # Load and merge market features
            market_features = self.load_market_features(start_date, end_date)
            for feature_name, feature_df in market_features.items():
                df = pd.merge_asof(
                    df.sort_values('time'),
                    feature_df,
                    on='time',
                    direction='backward'
                )
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Calculate fundamental features
            df = self.calculate_fundamental_features(df)
            
            # Calculate banking-specific features
            df = self.calculate_banking_features(df)
            
            # Create targets
            df = self.create_targets(df)
            
            # Clean data
            df = self._clean_data(df)
            
            logger.info(f"Processed {ticker}: {len(df)} rows, {len(self.feature_columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return None
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        df = df.copy()
        
        # Replace inf values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle negative values for ratio features
        ratio_features = ['Close_to_Open', 'High_to_Low']
        for feature in ratio_features:
            if feature in df.columns:
                df.loc[df[feature] < 0, feature] = np.nan
        
        # Fill NaN values with median
        for col in self.feature_columns:
            if col in df.columns:
                if df[col].isna().all():
                    logger.warning(f"Column {col} is entirely NaN, filling with 0")
                    df[col] = 0
                else:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        # Fill target NaN values
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        for col in target_cols:
            if col in df.columns:
                if 'Close' in col:
                    df[col] = df[col].fillna(df[col].median())
                else:  # Direction targets
                    df[col] = df[col].fillna(1)  # Default to flat
        
        # Remove rows where all targets are NaN
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        if target_cols:
            df = df.dropna(subset=target_cols, how='all')
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, timesteps: int = None) -> Dict[str, np.ndarray]:
        """Create sequences for deep learning models"""
        if timesteps is None:
            timesteps = self.config.get('training.timesteps', 30)
        
        if len(df) < timesteps:
            raise ValueError(f"Data length {len(df)} < timesteps {timesteps}")
        
        # Get available feature columns
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        # Exclude non-numeric columns
        numeric_features = []
        for col in available_features:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
        
        X = []
        targets = {}
        
        # Initialize target arrays
        target_cols = [col for col in df.columns if col.startswith('Target_')]
        for col in target_cols:
            targets[col] = []
        
        # Create sequences
        for i in range(len(df) - timesteps):
            # Features sequence
            X.append(df[numeric_features].iloc[i:i+timesteps].values)
            
            # Targets (single values at timestep i+timesteps)
            for col in target_cols:
                targets[col].append(df[col].iloc[i+timesteps])
        
        result = {'X': np.array(X)}
        for col, values in targets.items():
            result[col] = np.array(values)
        
        logger.info(f"Created sequences: X shape {result['X'].shape}, {len(target_cols)} targets, {len(numeric_features)} features")
        return result
    
    def scale_features(self, df: pd.DataFrame, scaler_type: str = None, 
                      fit_scaler: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Scale features and return scalers"""
        if scaler_type is None:
            scaler_type = self.config.get('training.scaler_type', 'minmax')
        
        df = df.copy()
        scalers = {}
        
        # Feature scaler
        if scaler_type == 'robust':
            feature_scaler = RobustScaler()
        elif scaler_type == 'standard':
            feature_scaler = StandardScaler()
        else:
            feature_scaler = MinMaxScaler()
        
        # Get available numeric features
        available_features = []
        for col in self.feature_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                available_features.append(col)
        
        if available_features:
            if fit_scaler:
                df[available_features] = feature_scaler.fit_transform(df[available_features])
                scalers['features'] = feature_scaler
            else:
                df[available_features] = feature_scaler.transform(df[available_features])
        
        # Target scalers (for regression targets)
        target_cols = [col for col in df.columns if col.startswith('Target_Close')]
        for col in target_cols:
            if col in df.columns:
                target_scaler = MinMaxScaler()
                if fit_scaler:
                    df[[col]] = target_scaler.fit_transform(df[[col]])
                    scalers[col] = target_scaler
                else:
                    df[[col]] = target_scaler.transform(df[[col]])
        
        return df, scalers
    
    def process_all_tickers(self, tickers: List[str] = None, 
                           start_date: str = None, end_date: str = None) -> Dict[str, bool]:
        """Process features for all tickers"""
        if tickers is None:
            tickers = self.config.tickers
        if start_date is None:
            start_date = self.config.get('data.start_date')
        if end_date is None:
            # Always use yesterday to ensure data availability
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"Processing features for {len(tickers)} tickers")
        
        results = {}
        timesteps = self.config.get('training.timesteps', 30)
        
        for ticker in tickers:
            try:
                # Process features
                df = self.process_ticker(ticker, start_date, end_date)
                if df is None:
                    results[ticker] = False
                    continue
                
                # Scale features
                df_scaled, scalers = self.scale_features(df, fit_scaler=True)
                
                # Save processed features
                features_path = os.path.join(self.config.processed_dir, f"{ticker}_features.csv")
                os.makedirs(self.config.processed_dir, exist_ok=True)
                df_scaled.to_csv(features_path, index=False)
                
                # Save scalers
                scalers_path = os.path.join(self.config.processed_dir, f"{ticker}_scalers.pkl")
                joblib.dump(scalers, scalers_path)
                
                # Create and save sequences
                sequences = self.create_sequences(df_scaled, timesteps)
                sequences_path = os.path.join(self.config.processed_dir, f"{ticker}_sequences.npz")
                np.savez_compressed(sequences_path, **sequences)
                
                results[ticker] = True
                logger.info(f"Successfully processed {ticker}")
                
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                results[ticker] = False
        
        successful = sum(results.values())
        logger.info(f"Feature processing completed: {successful}/{len(tickers)} successful")
        return results

def main():
    """Main function for feature engineering"""
    engineer = FeatureEngineer()
    results = engineer.process_all_tickers()
    
    successful = [ticker for ticker, success in results.items() if success]
    failed = [ticker for ticker, success in results.items() if not success]
    
    print(f"\n=== Feature Engineering Summary ===")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import joblib
import torch
from datetime import datetime, timedelta
import ta

# Add src to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import get_config
from src.features.feature_engineer import FeatureEngineer
from src.models.cnn_bilstm import CNNBiLSTM
from src.models.transformer import TransformerModel
from src.utils.database import get_database

# --- Page Configuration ---
st.set_page_config(
    page_title="Market Monitor & Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
CONFIG = get_config()
RAW_DATA_DIR = CONFIG.get('data.raw_dir', 'data/raw')
PROCESSED_DATA_DIR = CONFIG.get('data.processed_dir', 'data/processed')
MODEL_DIR = CONFIG.get('paths.models', 'models')
TICKERS = CONFIG.get('data.tickers', [])
MODELS_TO_TRAIN = CONFIG.get('models_to_train', {}).get('dl_models', [])
HORIZONS = CONFIG.get('models.shared.forecast_horizons', [])
SEQUENCE_LENGTH = CONFIG.get('models.shared.sequence_length', 30)

# --- Caching Functions ---

@st.cache_data(ttl=3600)  # Cache raw data for 1 hour
def load_raw_data(ticker):
    """Loads raw OHLCV data for a given ticker."""
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}_ohlcv.csv")
    if not os.path.exists(file_path):
        st.error(f"Raw data file not found for {ticker} at {file_path}. Please run data collection.")
        return None
    df = pd.read_csv(file_path, parse_dates=['time'])
    df = df.sort_values('time').reset_index(drop=True)
    return df

@st.cache_data
def calculate_technical_indicators(df):
    """Calculates technical indicators using the 'ta' library."""
    df_with_ta = df.copy()
    df_with_ta['EMA_34'] = ta.trend.ema_indicator(df_with_ta['Close'], window=34)
    df_with_ta['EMA_89'] = ta.trend.ema_indicator(df_with_ta['Close'], window=89)
    bollinger = ta.volatility.BollingerBands(df_with_ta['Close'])
    df_with_ta['BB_High'] = bollinger.bollinger_hband()
    df_with_ta['BB_Low'] = bollinger.bollinger_lband()
    df_with_ta['RSI'] = ta.momentum.rsi(df_with_ta['Close'])
    macd = ta.trend.MACD(df_with_ta['Close'])
    df_with_ta['MACD'] = macd.macd()
    df_with_ta['MACD_Signal'] = macd.macd_signal()
    return df_with_ta

@st.cache_resource
def load_model(model_type, model_path, input_dim):
    """Loads a PyTorch model and its weights."""
    if model_type.lower() == 'cnn_bilstm':
        model = CNNBiLSTM(input_dim=input_dim, config=CONFIG.get('models'))
    elif model_type.lower() == 'transformer':
        model = TransformerModel(input_dim=input_dim, config=CONFIG.get('models'))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None

@st.cache_data
def get_prediction_data(ticker, horizon):
    """Loads and prepares data required for AI prediction."""
    # 1. Load pre-generated metadata which includes scaler and feature columns
    metadata_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_metadata_t+{horizon}.pkl")
    if not os.path.exists(metadata_path):
        return None, "Metadata file not found. Please run feature engineering."

    metadata = joblib.load(metadata_path)
    scaler = metadata['scaler']
    feature_columns = metadata['feature_columns']

    # 2. Load raw data and generate features
    raw_df = load_raw_data(ticker)
    if raw_df is None or raw_df.empty:
        return None, "Could not load raw data."
    
    # Restrict to last N days for performance, ensuring enough history for long-term features
    raw_df = raw_df.tail(SEQUENCE_LENGTH + 380).copy()


    # 3. Instantiate feature engineer and replicate processing steps
    feature_engineer = FeatureEngineer()
    db = get_database()

    # 3a. Technical indicators
    processed_df = feature_engineer.calculate_technical_indicators(raw_df)
    
    # 3b. Fundamental data
    fundamental_data = db.load_dataframe(f"{ticker}_Fundamental")
    if fundamental_data is not None and not fundamental_data.empty:
        fundamental_data['time'] = pd.to_datetime(fundamental_data['time'])
        fundamental_data = fundamental_data.sort_values('time').drop_duplicates(subset='time', keep='last')
        processed_df = pd.merge_asof(processed_df, fundamental_data, on='time', direction='backward')
        processed_df = feature_engineer.calculate_banking_features(processed_df, fundamental_data)
    else:
        processed_df = feature_engineer.calculate_banking_features(processed_df, None)

    # 3c. Market data
    vnindex_path = os.path.join(CONFIG.get('data.raw_dir', 'data/raw'), "VNINDEX.csv")
    if os.path.exists(vnindex_path):
        market_df = pd.read_csv(vnindex_path, parse_dates=['time'])
        market_df['Market_Pct_Change'] = market_df.get('VNINDEX', pd.Series(dtype=float)).pct_change()
        market_df['Market_Volatility'] = market_df.get('VNINDEX', pd.Series(dtype=float)).rolling(window=14).std()
        market_df = market_df[['time', 'Market_Pct_Change', 'Market_Volatility']]
        processed_df = pd.merge(processed_df, market_df, on='time', how='left')


    # 4. Clean data: Ensure all features exist and fill NaNs
    for col in feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0 # Add missing columns with 0
            
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    processed_df[feature_columns] = processed_df[feature_columns].ffill().bfill()
    processed_df[feature_columns] = processed_df[feature_columns].fillna(0)

    # 5. Scale the features and take the last sequence
    if not all(col in processed_df.columns for col in feature_columns):
        missing_cols = [c for c in feature_columns if c not in processed_df.columns]
        return None, f"Mismatched columns after feature engineering. Missing: {missing_cols}"
        
    data_to_scale = processed_df[feature_columns]
    
    if data_to_scale.isnull().values.any():
         return None, "NaN values found in feature data before scaling. Check cleaning logic."

    scaled_features = scaler.transform(data_to_scale)
    
    if len(scaled_features) < SEQUENCE_LENGTH:
        return None, f"Not enough data to form a sequence of {SEQUENCE_LENGTH} days."
        
    latest_sequence = scaled_features[-SEQUENCE_LENGTH:]
    latest_sequence_tensor = torch.FloatTensor(latest_sequence).unsqueeze(0) # Add batch dimension

    return latest_sequence_tensor, None

# --- UI Layout ---

st.title("📈 Market Monitor & Forecast Dashboard")

# --- Sidebar ---
st.sidebar.title("Configuration")
st.sidebar.markdown("Select the stock, model, and forecast horizon.")

selected_ticker = st.sidebar.selectbox("Stock Symbol", TICKERS, index=TICKERS.index('VCB') if 'VCB' in TICKERS else 0)
selected_model = st.sidebar.selectbox("AI Model", MODELS_TO_TRAIN, index=MODELS_TO_TRAIN.index('cnn_bilstm') if 'cnn_bilstm' in MODELS_TO_TRAIN else 0)
selected_horizon = st.sidebar.selectbox("Forecast Horizon (Days)", HORIZONS, index=HORIZONS.index(30) if 30 in HORIZONS else 0)

st.sidebar.markdown("---")
# Date range selector for the historical chart
today = datetime.today()
default_start = today - timedelta(days=365)
date_range = st.sidebar.date_input(
    "Select Date Range for Chart",
    (default_start, today),
    min_value=datetime(2010, 1, 1),
    max_value=today,
    format="YYYY-MM-DD"
)
if len(date_range) != 2:
    st.sidebar.warning("Please select a valid start and end date.")
    st.stop()
start_date, end_date = date_range

# --- Main Content Tabs ---
tab1, tab2 = st.tabs(["Market Monitor", "🔮 AI Forecast"])

# --- Tab 1: Market Monitor ---
with tab1:
    st.header(f"Real-time Market Data for {selected_ticker}")
    df_raw = load_raw_data(selected_ticker)
    
    if df_raw is not None:
        df_ta = calculate_technical_indicators(df_raw)
        latest_data = df_ta.iloc[-1]

        # --- Metrics Row ---
        price = latest_data['Close']
        prev_price = df_ta.iloc[-2]['Close']
        price_change = ((price - prev_price) / prev_price) * 100
        macd_status = "Bullish" if latest_data['MACD'] > latest_data['MACD_Signal'] else "Bearish"

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Current Price", f"{price:,.0f} VND", f"{price_change:.2f}%")
        m_col2.metric("RSI (14)", f"{latest_data['RSI']:.2f}")
        m_col3.metric("MACD Status", macd_status)
        m_col4.metric("Volume", f"{latest_data['Volume']:,.0f}")
        
        # --- Chart ---
        df_filtered = df_ta[(df_ta['time'].dt.date >= start_date) & (df_ta['time'].dt.date <= end_date)]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{selected_ticker} Candlestick Chart', 'Volume'),
            row_heights=[0.75, 0.25]
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_filtered['time'], open=df_filtered['Open'], high=df_filtered['High'],
            low=df_filtered['Low'], close=df_filtered['Close'], name='OHLC'
        ), row=1, col=1)

        # Indicators
        fig.add_trace(go.Scatter(x=df_filtered['time'], y=df_filtered['EMA_34'], name='EMA 34', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=df_filtered['time'], y=df_filtered['EMA_89'], name='EMA 89', line=dict(color='purple', width=1)))
        fig.add_trace(go.Scatter(x=df_filtered['time'], y=df_filtered['BB_High'], name='Bollinger High', line=dict(color='gray', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df_filtered['time'], y=df_filtered['BB_Low'], name='Bollinger Low', line=dict(color='gray', width=1, dash='dash')))

        # Volume
        fig.add_trace(go.Bar(x=df_filtered['time'], y=df_filtered['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)

        fig.update_layout(
            height=700,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price (VND)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: AI Forecast ---
with tab2:
    st.header(f"AI-Powered Forecast for {selected_ticker}")

    # --- Prediction Pipeline ---
    model_name_key = f"{selected_ticker}_{selected_model}_t+{selected_horizon}"
    model_filename = f"{model_name_key}.pt"
    model_path = os.path.join(MODEL_DIR, model_filename)

    st.info(f"Attempting to predict trend for **{selected_ticker}** over the next **{selected_horizon} days** using the **{selected_model.upper()}** model.")

    # Prepare data for prediction
    with st.spinner("Preparing data and running inference..."):
        prediction_tensor, error = get_prediction_data(selected_ticker, selected_horizon)
    
    if error:
        st.error(f"Data Preparation Error: {error}")
    elif prediction_tensor is None:
        st.error("Unknown error during data preparation.")
    else:
        # Load model and make prediction
        # Get input_dim from the loaded data tensor
        input_dim = prediction_tensor.shape[2] 
        model = load_model(selected_model, model_path, input_dim)

        if model is None:
            st.error(f"Model file not found at '{model_path}'. Please ensure the model is trained and the file exists.")
        else:
            with torch.no_grad():
                output = model(prediction_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze().tolist()
                
            # Assuming class 0 = Down, class 1 = Up
            prob_down, prob_up = probabilities[0], probabilities[1]
            predicted_class = np.argmax(probabilities)
            confidence = max(probabilities)
            
            trend = "Up" if predicted_class == 1 else "Down"
            trend_emoji = "🔼" if trend == "Up" else "🔽"

            # --- Display Prediction ---
            st.subheader("Prediction Result")
            p_col1, p_col2 = st.columns(2)
            
            with p_col1:
                st.metric(
                    f"Predicted Trend (t+{selected_horizon} days)",
                    f"{trend} {trend_emoji}",
                    help=f"Model confidence in this prediction is {confidence:.2%}"
                )
            with p_col2:
                st.write("**Prediction Confidence:**")
                st.progress(prob_up, text=f"Up: {prob_up:.1%}")
                st.progress(prob_down, text=f"Down: {prob_down:.1%}")

            # --- Actionable Insight ---
            st.subheader("Actionable Insight")
            insight_text = (
                f"The **{selected_model.upper()}** model predicts an **{trend.upper()}** trend for **{selected_ticker}** "
                f"in the next **{selected_horizon} days** with **{confidence:.1%}** confidence. "
            )
            if trend == "Up" and confidence > 0.7:
                insight_text += "This indicates a strong bullish signal."
            elif trend == "Down" and confidence > 0.7:
                insight_text += "This indicates a strong bearish signal."
            else:
                insight_text += "This signal suggests the indicated trend, but with moderate confidence."
            st.success(insight_text)

            # --- Historical vs. Predicted Chart ---
            st.subheader("Historical Price vs. Forecast Point")
            df_hist = load_raw_data(selected_ticker).tail(90)
            
            if df_hist is not None:
                last_price = df_hist['Close'].iloc[-1]
                last_date = df_hist['time'].iloc[-1]
                
                # Simple price projection based on threshold
                threshold = CONFIG['features']['dynamic_neutral_thresholds']['short_term']['threshold']
                if selected_horizon >= 30:
                    threshold = CONFIG['features']['dynamic_neutral_thresholds']['mid_term']['threshold']
                if selected_horizon >= 60:
                    threshold = CONFIG['features']['dynamic_neutral_thresholds']['long_term']['threshold']

                price_change_factor = (1 + threshold) if trend == "Up" else (1 - threshold)
                predicted_price = last_price * price_change_factor
                predicted_date = last_date + timedelta(days=selected_horizon)

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=df_hist['time'], y=df_hist['Close'],
                    mode='lines', name='Historical Price', line=dict(color='royalblue')
                ))
                fig_pred.add_trace(go.Scatter(
                    x=[last_date, predicted_date],
                    y=[last_price, predicted_price],
                    mode='lines+markers', name='Forecast',
                    line=dict(color='orange', dash='dot'),
                    marker=dict(size=8)
                ))
                
                fig_pred.update_layout(
                    title=f"90-Day History and t+{selected_horizon} Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (VND)",
                    height=500
                )
                st.plotly_chart(fig_pred, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "**Disclaimer:** This application is for educational and research purposes only. "
    "Stock predictions are inherently uncertain and should not be the sole basis for investment decisions."
)

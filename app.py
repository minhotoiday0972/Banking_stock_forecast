# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import get_config
from src.app.predictor import StockPredictor

# Page config
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

@st.cache_resource
def init_app():
    try:
        config = get_config()
        predictor = StockPredictor()
        return config, predictor
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return None, None

def main():
    config, predictor = init_app()
    if config is None or predictor is None:
        st.stop()
    
    st.title("🏦 Vietnamese Banking Stock Trend Predictor")
    st.markdown("Dự đoán xu hướng cổ phiếu ngành ngân hàng Việt Nam bằng các mô hình học sâu chuyên biệt.")
    
    # --- Sidebar ---
    st.sidebar.header("Cài đặt Dự báo")
    
    tickers = config.get('data.tickers', [])
    ticker = st.sidebar.selectbox("Chọn Ngân hàng", tickers)
    
    available_models = predictor.get_available_models(ticker)
    if not available_models:
        st.error(f"Không tìm thấy model đã huấn luyện cho mã {ticker}. Vui lòng chạy 'python main.py train --models all --tickers {ticker}'")
        st.stop()
    
    model_type = st.sidebar.selectbox("Chọn Model", available_models)
    
    # --- THAY ĐỔI: Tầm nhìn mới ---
    horizons = config.get('models.shared.forecast_horizons', [1, 3, 5, 30, 60, 90])
    horizon = st.sidebar.selectbox(
        "Chu kỳ Dự báo (ngày)",
        horizons,
        help="Chọn tầm nhìn dự báo. Mỗi tầm nhìn sử dụng một mô hình chuyên biệt."
    )
    
    if st.sidebar.button("Thực hiện Dự báo", type="primary"):
        with st.spinner(f"Đang chạy mô hình {model_type.upper()} cho t+{horizon} ngày..."):
            prediction = predictor.predict(ticker, model_type, horizon)
            st.session_state['prediction'] = prediction
            
    # --- Hiển thị Kết quả Dự báo ---
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        
        # Đảm bảo kết quả là của lựa chọn hiện tại
        if prediction and prediction['ticker'] == ticker and prediction['model_type'] == model_type and prediction['horizon'] == horizon:
            st.header("📊 Kết quả Dự báo")
            
            # --- THAY ĐỔI: Chỉ còn 2 cột ---
            col1, col2 = st.columns(2)
            
            with col1:
                direction = prediction['predicted_direction']
                confidence = prediction['direction_confidence']
                direction_emoji = {'Up': '🔼', 'Down': '🔽', 'Neutral': '↔️'}.get(direction, '❓')
                
                st.metric(
                    f"Xu hướng Dự báo (t+{horizon} ngày)",
                    f"{direction} {direction_emoji}",
                    help=f"Độ tin cậy: {confidence:.2%}"
                )
            
            with col2:
                st.metric(
                    "Model sử dụng",
                    model_type.upper(),
                    help=f"Mô hình này được huấn luyện chuyên biệt cho tầm nhìn t+{horizon} ngày."
                )
            
            # Hiển thị xác suất cho cả 3 lớp
            st.progress(prediction['direction_probabilities'].get('Up', 0), text=f"Xác suất Tăng (Up): {prediction['direction_probabilities'].get('Up', 0):.1%}")
            st.progress(prediction['direction_probabilities'].get('Neutral', 0), text=f"Xác suất Đi ngang (Neutral): {prediction['direction_probabilities'].get('Neutral', 0):.1%}")
            st.progress(prediction['direction_probabilities'].get('Down', 0), text=f"Xác suất Giảm (Down): {prediction['direction_probabilities'].get('Down', 0):.1%}")

    # --- THÊM MỚI: Hiển thị Dữ liệu Cơ bản ---
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Dữ liệu Cơ bản (Mới nhất)")
    fundamentals = predictor.get_latest_fundamentals(ticker)
    if fundamentals is not None:
        st.sidebar.caption(f"Dữ liệu được ghi nhận vào: {fundamentals.get('time', 'N/A')}")
        
        # Chọn các chỉ số quan trọng để hiển thị
        kpi_map = {
            'NIM (%)': 'NIM',
            'NPL (%)': 'NPL (Nợ xấu)',
            'CIR (%)': 'CIR (Chi phí/Thu nhập)',
            'ROE (%)': 'ROE',
            'P/E': 'P/E',
            'P/B': 'P/B',
            'Credit_Growth (%)': 'Tăng trưởng Tín dụng'
        }
        
        for key, label in kpi_map.items():
            if key in fundamentals:
                value = fundamentals[key]
                if isinstance(value, (float, np.floating)):
                    st.sidebar.metric(label, f"{value:.2f} %" if "%" in key else f"{value:.2f}")

    # --- Biểu đồ Lịch sử Giá (Giữ nguyên) ---
    st.header("📈 Biểu đồ Lịch sử Giá")
    historical_data = predictor.get_historical_data(ticker, days=180) # Lấy nhiều ngày hơn
    if historical_data is not None and not historical_data.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f'Biểu đồ giá {ticker}', 'Khối lượng Giao dịch'), row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=historical_data['time'], open=historical_data['Open'], high=historical_data['High'], low=historical_data['Low'], close=historical_data['Close'], name='Giá'), row=1, col=1)
        fig.add_trace(go.Bar(x=historical_data['time'], y=historical_data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
        fig.update_layout(xaxis_title="Ngày", yaxis_title="Giá (VND)", height=600, showlegend=False)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Không có dữ liệu lịch sử cho mã {ticker}")
    
    # --- Footer (Giữ nguyên) ---
    st.markdown("---")
    st.markdown("""
    **Miễn trừ trách nhiệm:** Ứng dụng này chỉ dành cho mục đích học tập và nghiên cứu. 
    Các dự đoán về chứng khoán vốn không chắc chắn và không nên được sử dụng làm cơ sở duy nhất cho các quyết định đầu tư.
    """)

if __name__ == "__main__":
    main()
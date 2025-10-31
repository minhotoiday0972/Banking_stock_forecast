# app.py - Main Streamlit application entry point
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.app.predictor import StockPredictor

# Page config
st.set_page_config(
    page_title="Vietnamese Banking Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize
@st.cache_resource
def init_app():
    """Initialize application components"""
    try:
        config = get_config()
        predictor = StockPredictor()
        return config, predictor
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return None, None

def main():
    """Main Streamlit application"""
    
    config, predictor = init_app()
    if config is None or predictor is None:
        st.stop()
    
    st.title("🏦 Vietnamese Banking Stock Predictor")
    st.markdown("""
    Dự đoán xu hướng cổ phiếu ngành ngân hàng Việt Nam bằng các mô hình học sâu.
    Chọn một ngân hàng, mô hình và chu kỳ dự báo để nhận kết quả.
    """)
    
    st.sidebar.header("Cài đặt Dự báo")
    
    tickers = config.get('data.tickers', [])
    ticker = st.sidebar.selectbox("Chọn Ngân hàng", tickers)
    
    available_models = predictor.get_available_models(ticker)
    if not available_models:
        st.error(f"Không tìm thấy model đã huấn luyện cho mã {ticker}. Vui lòng huấn luyện trước.")
        st.code(f"python main.py train --models all --tickers {ticker}")
        st.stop()
    
    model_type = st.sidebar.selectbox("Chọn Model", available_models)
    horizons = config.get('models.shared.forecast_horizons', [1, 3, 5])
    horizon = st.sidebar.selectbox("Chu kỳ Dự báo (ngày)", horizons)
    
    if st.sidebar.button("Thực hiện Dự báo", type="primary"):
        with st.spinner("Đang thực hiện dự báo..."):
            prediction = predictor.predict(ticker, model_type, horizon)
        
        if prediction is None:
            st.error("Thực hiện dự báo thất bại. Vui lòng kiểm tra logs để biết chi tiết.")
            st.stop()

        st.header("📊 Kết quả Dự báo")
        st.session_state['prediction'] = prediction

    # Display prediction results if they exist in session state
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']
        
        # Check if the prediction is for the currently selected options
        if prediction.get('ticker') == ticker and prediction.get('model_type') == model_type and prediction.get('horizon') == horizon:
            col1, col2, col3 = st.columns(3)
            with col1:
                direction = prediction.get('predicted_direction', 'N/A')
                confidence = prediction.get('direction_confidence', 0)
                direction_emoji = {'Up': '🔼', 'Down': '🔽'}.get(direction, '↔️')
                st.metric("Xu hướng Dự báo", f"{direction} {direction_emoji}", help=f"Độ tin cậy: {confidence:.2%}")

            with col2:
                predicted_price = prediction.get('predicted_price')
                if predicted_price is not None:
                    metric_delta = None
                    metric_help = f"Giá dự báo cho {horizon} ngày tới"
                    if 'current_price' in prediction and 'price_change_pct' in prediction:
                        price_change_pct = prediction['price_change_pct']
                        metric_delta = f"{price_change_pct:+.2%}"
                        metric_help = f"Hiện tại: {prediction['current_price']:,.0f} VND → Dự báo: {predicted_price:,.0f} VND"
                    
                    st.metric("Giá Dự báo", f"{predicted_price:,.0f} VND", delta=metric_delta, help=metric_help)
            
            with col3:
                st.metric("Model sử dụng", model_type.upper(), help=f"Chu kỳ dự báo: {horizon} ngày")

    # Historical data visualization
    st.header("📈 Biểu đồ Lịch sử Giá")
    historical_data = predictor.get_historical_data(ticker, days=90)
    if historical_data is not None and not historical_data.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=(f'Biểu đồ giá {ticker}', 'Khối lượng Giao dịch'), row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=historical_data['time'], open=historical_data['Open'], high=historical_data['High'], low=historical_data['Low'], close=historical_data['Close'], name='Giá'), row=1, col=1)
        fig.add_trace(go.Bar(x=historical_data['time'], y=historical_data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
        fig.update_layout(title_text=f"{ticker} - Dữ liệu 90 ngày gần nhất", xaxis_title="Ngày", yaxis_title="Giá (VND)", height=600, showlegend=False)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Không có dữ liệu lịch sử cho mã {ticker}")
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.info("Thông tin về các mô hình được sử dụng trong dự án.")
    st.sidebar.expander("CNN-BiLSTM").write("Kết hợp Mạng Tích chập (CNN) để nhận dạng mẫu và Mạng LSTM Hai chiều (BiLSTM) để mô hình hóa chuỗi thời gian. Mạnh trong việc nhận dạng mẫu cục bộ và nắm bắt các phụ thuộc dài hạn.")
    st.sidebar.expander("Transformer").write("Sử dụng cơ chế tự chú ý (self-attention) để mô hình hóa các mối quan hệ phức tạp trong dữ liệu. Vượt trội với chuỗi dữ liệu dài và nắm bắt các mẫu hình phức tạp.")

if __name__ == "__main__":
    main()
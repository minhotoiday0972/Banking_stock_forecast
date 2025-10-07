# src/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path to access src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.app.predictor import StockPredictor

# Page config
st.set_page_config(
    page_title="Vietnamese Banking Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
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

    # Initialize
    config, predictor = init_app()
    if config is None or predictor is None:
        st.stop()

    # Title and description
    st.title("🏦 Vietnamese Banking Stock Predictor")
    st.markdown(
        """
    Predict Vietnamese banking stock trends using advanced deep learning models.
    Choose a bank, model, and forecast horizon to get predictions.
    """
    )

    # Sidebar controls
    st.sidebar.header("Prediction Settings")

    # Ticker selection
    tickers = config.tickers
    ticker = st.sidebar.selectbox(
        "Select Bank", tickers, help="Choose the banking stock to predict"
    )

    # Check available models for selected ticker
    available_models = predictor.get_available_models(ticker)

    if not available_models:
        st.error(f"No trained models found for {ticker}. Please train models first.")
        st.stop()

    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model", available_models, help="Choose the prediction model"
    )

    # Horizon selection
    horizons = config.get("models.forecast_horizons", [1, 3, 5])
    horizon = st.sidebar.selectbox(
        "Forecast Horizon (days)", horizons, help="Number of days ahead to predict"
    )

    # Prediction button
    if st.sidebar.button("Make Prediction", type="primary"):
        with st.spinner("Making prediction..."):
            prediction = predictor.predict(ticker, model_type, horizon)

        if prediction is None:
            st.error("Failed to make prediction. Please check logs.")
            return

        # Display prediction results
        st.header("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            if "predicted_direction" in prediction:
                direction = prediction["predicted_direction"]
                confidence = prediction["direction_confidence"]

                # Color based on direction
                color = {"Up": "green", "Down": "red", "Flat": "orange"}.get(
                    direction, "gray"
                )

                st.metric(
                    "Predicted Direction",
                    direction,
                    help=f"Confidence: {confidence:.2%}",
                )

                # Direction probabilities
                if "direction_probabilities" in prediction:
                    probs = prediction["direction_probabilities"]
                    st.write("**Direction Probabilities:**")
                    for label, prob in probs.items():
                        st.write(f"- {label}: {prob:.2%}")

        with col2:
            if "predicted_price" in prediction:
                st.metric(
                    "Predicted Price",
                    f"{prediction['predicted_price']:.2f} VND",
                    help=f"Price prediction for {horizon} days ahead",
                )

        with col3:
            st.metric(
                "Model Used",
                model_type.upper(),
                help=f"Forecast horizon: {horizon} days",
            )

    # Historical data visualization
    st.header("📈 Historical Price Data")

    # Get historical data
    historical_data = predictor.get_historical_data(ticker, days=90)

    if historical_data is not None and not historical_data.empty:
        # Create candlestick chart
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f"{ticker} Price Chart", "Volume"),
            row_width=[0.7, 0.3],
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=historical_data["time"],
                open=historical_data["Open"],
                high=historical_data["High"],
                low=historical_data["Low"],
                close=historical_data["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Volume chart
        fig.add_trace(
            go.Bar(
                x=historical_data["time"],
                y=historical_data["Volume"],
                name="Volume",
                marker_color="lightblue",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=f"{ticker} - Last 90 Days",
            xaxis_title="Date",
            yaxis_title="Price (VND)",
            height=600,
            showlegend=False,
        )

        fig.update_xaxes(rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

        # Recent price statistics
        st.subheader("📊 Recent Statistics")

        col1, col2, col3, col4 = st.columns(4)

        latest_price = historical_data["Close"].iloc[-1]
        price_change = (
            historical_data["Close"].iloc[-1] - historical_data["Close"].iloc[-2]
        )
        price_change_pct = (price_change / historical_data["Close"].iloc[-2]) * 100

        with col1:
            st.metric("Latest Price", f"{latest_price:.2f} VND")

        with col2:
            st.metric(
                "Daily Change", f"{price_change:+.2f} VND", f"{price_change_pct:+.2f}%"
            )

        with col3:
            high_52w = historical_data["High"].max()
            st.metric("90-Day High", f"{high_52w:.2f} VND")

        with col4:
            low_52w = historical_data["Low"].min()
            st.metric("90-Day Low", f"{low_52w:.2f} VND")

        # Price distribution
        st.subheader("📊 Price Distribution")

        fig_hist = px.histogram(
            historical_data,
            x="Close",
            nbins=20,
            title=f"{ticker} Price Distribution (Last 90 Days)",
        )
        fig_hist.update_layout(xaxis_title="Price (VND)", yaxis_title="Frequency")
        st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.warning(f"No historical data available for {ticker}")

    # Model information
    st.header("🤖 Model Information")

    model_info = {
        "cnn_bilstm": {
            "name": "CNN-BiLSTM",
            "description": "Combines Convolutional Neural Networks with Bidirectional LSTM for pattern recognition and sequence modeling.",
            "strengths": [
                "Good for local pattern detection",
                "Captures long-term dependencies",
                "Robust performance",
            ],
        },
        "transformer": {
            "name": "Transformer",
            "description": "Uses self-attention mechanism to model complex temporal relationships in stock data.",
            "strengths": [
                "Excellent for long sequences",
                "Captures complex patterns",
                "State-of-the-art architecture",
            ],
        },
        "lstm": {
            "name": "LSTM",
            "description": "Long Short-Term Memory network designed for sequence prediction tasks.",
            "strengths": [
                "Reliable baseline",
                "Good interpretability",
                "Efficient training",
            ],
        },
    }

    if model_type in model_info:
        info = model_info[model_type]
        st.subheader(f"📋 {info['name']}")
        st.write(info["description"])
        st.write("**Strengths:**")
        for strength in info["strengths"]:
            st.write(f"- {strength}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    **Disclaimer:** This application is for educational and research purposes only. 
    Stock predictions are inherently uncertain and should not be used as the sole basis for investment decisions.
    Always consult with financial professionals before making investment choices.
    """
    )


if __name__ == "__main__":
    main()

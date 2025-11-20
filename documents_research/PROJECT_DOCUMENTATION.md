# TÀI LIỆU DỰ ÁN - HỆ THỐNG DỰ ĐOÁN GIÁ CỔ PHIẾU NGÂN HÀNG VIỆT NAM

> **Lưu ý**: Tài liệu này được tạo dựa trên code thực tế của dự án (2025-11-11)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mục Tiêu
Xây dựng hệ thống dự đoán **hướng biến động giá** (tăng/giảm) cổ phiếu ngân hàng Việt Nam sử dụng Deep Learning.

### 1.2 Phạm Vi
- **11 ngân hàng**: VIB, VCB, BID, MBB, TCB, VPB, CTG, ACB, SHB, STB, HDB
- **Thời gian**: 2020-01-01 đến hiện tại
- **Tần suất**: Dữ liệu hàng ngày
- **Horizons dự đoán**: 1, 3, 5, 30, 60, 90 ngày

### 1.3 Loại Bài Toán
**Binary Classification** (2 classes):
- Class 0: Down (giá giảm hoặc không đổi, ≤ 0%)
- Class 1: Up (giá tăng, > 0%)

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1 Data Pipeline
```
Data Collection (vnstock API)
    ↓
Feature Engineering (Technical + Banking + Temporal)
    ↓
Feature Selection (RF-based, per horizon)
    ↓
Scaling (MinMaxScaler, fit on train proxy)
    ↓
Sequence Creation (30-day windows)
    ↓
Model Training (per ticker, per model, per horizon)
    ↓
Evaluation & Deployment
```

### 2.2 Models
1. **CNN-BiLSTM**: Hybrid architecture (CNN cho local patterns, BiLSTM cho temporal)
2. **Transformer**: Attention-based architecture (Multi-head attention)

### 2.3 Tech Stack
- **Framework**: PyTorch
- **Data**: vnstock, pandas, numpy
- **Database**: SQLite
- **Web**: Streamlit
- **Visualization**: Plotly, matplotlib

---

## 3. THU THẬP DỮ LIỆU

### 3.1 Nguồn Dữ Liệu
```python
# OHLCV Data
- Source: vnstock API (TCBS)
- Frequency: Daily
- Fields: Open, High, Low, Close, Volume

# Fundamental Data
- Source: vnstock API
- Frequency: Quarterly
- Fields: NIM, NPL, CIR, ROE, ROA, Credit Growth, etc.

# Market Data
- Source: VNINDEX CSV file
- Fields: Market index, volatility
```

### 3.2 Rate Limiting
```yaml
batch_size: 3                    # 3 tickers per batch
delay_between_tickers: 5s        # Delay between tickers
delay_between_batches: 10s       # Delay between batches
delay_between_requests: 2s       # Delay between API calls
```

### 3.3 Data Validation
- Giá > 0 (Open, High, Low, Close)
- Volume ≥ 0
- Loại bỏ outliers và missing values
- Retry mechanism: 3 attempts với 5s wait

---

## 4. FEATURE ENGINEERING

### 4.1 Technical Indicators (Ngắn hạn)
```python
# Moving Averages
- Close_MA7, Close_MA14, Close_MA30

# Volatility
- Volatility_14 (14-day rolling std)

# Price Ratios
- Close_to_Open = (Close - Open) / Open
- High_to_Low = (High - Low) / Low
- Close_Pct_Change = daily returns

# Advanced Indicators
- RSI_14 (Relative Strength Index)
- MACD, MACD_Signal
- BB_Upper, BB_Lower (Bollinger Bands)
```

### 4.2 Technical Indicators (Dài hạn)
```python
# Long-term Moving Averages
- Close_MA100, Close_MA200

# Long-term Volatility
- Volatility_60 (3-month volatility)

# Long-term RSI
- RSI_30
```

### 4.3 Banking-Specific Features
```python
# Core Banking Ratios
- NIM (%) - Net Interest Margin
- NPL (%) - Non-Performing Loan ratio
- CIR (%) - Cost-to-Income Ratio
- Credit_Growth (%)
- ROE (%), ROA (%)
- Loan_to_Deposit (%)
- Equity_Ratio (%)

# Derived Features
- NIM_Diff, NPL_Diff, CIR_Diff, Credit_Growth_Diff
- NIM_CIR_Ratio
- NPL_Trend (quarterly trend)
```

### 4.4 Temporal Features
```python
# Time Index
- time_index (linear index)

# Calendar Features
- month (1-12)
- day_of_year (1-365)
- day_of_week (0-6)
- year

# Lag Features
- Close_lag_90 (giá 90 ngày trước)
- Close_lag_365 (giá 365 ngày trước)
```

### 4.5 Market Features
```python
- Market_Pct_Change (VNINDEX daily return)
- Market_Volatility (VNINDEX 14-day std)
```

### 4.6 Feature Selection Strategy
**Logic "Bộ lọc Thông minh"**:
```python
if horizon in [1, 3, 5]:  # Ngắn hạn
    # Chỉ dùng technical indicators + fundamental diffs
    candidate_features = SHORT_TERM_UNIVERSE
else:  # horizon in [30, 60, 90] - Dài hạn
    # Dùng tất cả features (bao gồm cả long-term indicators)
    candidate_features = LONG_TERM_UNIVERSE

# Random Forest Feature Selection
# Chọn top 20 features quan trọng nhất cho mỗi horizon
golden_features = RF_feature_importance(candidate_features, top_n=20)
```

**Lý do**:
- Ngắn hạn (1-5 ngày): Giá chủ yếu bị ảnh hưởng bởi technical factors
- Dài hạn (30-90 ngày): Cần cả fundamental và macro factors

---

## 5. DATA PREPROCESSING

### 5.1 Merge Strategy (Tránh Data Leakage)
```python
# Merge fundamental data
df = pd.merge_asof(
    ohlcv_data, 
    fundamental_data, 
    on='time', 
    direction='backward'  # ✅ CHỈ DÙNG DỮ LIỆU QUÁ KHỨ
)

# Merge banking features
df = pd.merge_asof(
    df, 
    quarterly_npl, 
    on='time', 
    direction='backward'  # ✅ CHỈ DÙNG DỮ LIỆU QUÁ KHỨ
)
```

### 5.2 Scaling Strategy (Tránh Data Leakage)
```python
# Fit scaler CHỈ trên 80% dữ liệu đầu (train proxy)
train_proxy_size = int(len(df) * 0.8)
df_train_proxy = df.iloc[:train_proxy_size]

scaler = MinMaxScaler()
scaler.fit(df_train_proxy[golden_features])  # ✅ CHỈ FIT TRÊN TRAIN

# Transform toàn bộ data
df[golden_features] = scaler.transform(df[golden_features])
```

### 5.3 Target Creation
```python
# Binary classification
for horizon in [1, 3, 5, 30, 60, 90]:
    future_price = df['Close'].shift(-horizon)
    price_change = (future_price - df['Close']) / df['Close']
    df[f'Target_Direction_t+{horizon}'] = np.where(price_change > 0, 1, 0)
    # 0: Down/Flat (≤ 0%), 1: Up (> 0%)
```

### 5.4 Data Cleaning
```python
# Replace inf/-inf with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Fill NaN cho features (KHÔNG fill cho targets)
for col in feature_columns:
    if df[col].isna().all():
        df[col] = 0
    else:
        df[col] = df[col].fillna(df[col].median())
```

---

## 6. MODEL ARCHITECTURE

### 6.1 CNN-BiLSTM Model

#### Architecture
```python
Input: (batch_size, timesteps=30, features)
    ↓
Transpose: (batch, features, timesteps)
    ↓
Conv1D(in=features, out=64, kernel=3, padding=1)
    ↓
ReLU
    ↓
MaxPool1D(kernel=2)
    ↓
Dropout(0.5)  # ← Regularization 1
    ↓
Transpose: (batch, timesteps, 64)
    ↓
BiLSTM(input=64, hidden=64, layers=2, dropout=0.5)  # ← Regularization 2
    ↓
Last Timestep: (batch, 128)  # 64*2 directions
    ↓
Dropout(0.5)  # ← Regularization 3
    ↓
Linear(128 → 2)  # Classification head
    ↓
Output: (batch, 2)  # Logits for 2 classes
```

#### Hyperparameters
```yaml
hidden_dim: 64
num_layers: 2
kernel_size: 3
dropout_rate: 0.5  # High dropout for financial data
```

#### Đặc điểm
- **CNN**: Trích xuất local patterns (3-day windows)
- **BiLSTM**: Học temporal dependencies (forward + backward)
- **Dropout cao (0.5)**: Cần thiết cho financial data (noisy)

### 6.2 Transformer Model

#### Architecture
```python
Input: (batch_size, timesteps=30, features)
    ↓
Linear Projection: (batch, timesteps, d_model=64)
    ↓
Positional Encoding (sinusoidal)
    ↓
Dropout(0.5)  # ← Regularization 1
    ↓
TransformerEncoderLayer × 2:
    ├── Multi-Head Attention (4 heads)
    ├── Feed Forward (dim=128)
    └── Dropout(0.5)  # ← Regularization 2
    ↓
Last Timestep: (batch, 64)
    ↓
Dropout(0.5)  # ← Regularization 3
    ↓
Linear(64 → 2)  # Classification head
    ↓
Output: (batch, 2)  # Logits for 2 classes
```

#### Hyperparameters
```yaml
d_model: 64
nhead: 4
num_layers: 2
dim_feedforward: 128
dropout_rate: 0.5
```

#### Đặc điểm
- **Positional Encoding**: Cung cấp thông tin vị trí thời gian
- **Multi-Head Attention**: Học được long-range dependencies
- **Parallel Processing**: Nhanh hơn RNN/LSTM

---

## 7. TRAINING PROCEDURE

### 7.1 Data Splitting
```python
train_split: 0.8  # 80% đầu cho training
val_split: 0.1    # 10% giữa cho validation
test_split: 0.1   # 10% cuối cho testing

# QUAN TRỌNG: Không shuffle, giữ nguyên thứ tự thời gian
```

### 7.2 Sequence Creation
```python
timesteps: 30  # 30 ngày lịch sử

# Sliding window
for i in range(len(data) - timesteps + 1):
    X[i] = data[i:i+timesteps]  # 30 ngày lịch sử
    y[i] = target[i+timesteps-1]  # Target tại ngày cuối
```

### 7.3 Training Configuration
```yaml
epochs: 100
batch_size: 32
learning_rate: 0.001  # Initial LR
early_stopping_patience: 25
```

### 7.4 Loss Function
```python
# CrossEntropyLoss với class weights
class_weights = calculate_class_weights(train_targets)
# weights = total_samples / (n_classes * class_counts)

criterion = nn.CrossEntropyLoss(weight=class_weights)
loss = criterion(outputs, targets)
```

### 7.5 Optimizer & Scheduler
```python
# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ReduceLROnPlateau scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,        # Giảm LR 10 lần
    patience=10,       # Chờ 10 epochs
    min_lr=1e-7
)

# Gọi mỗi epoch
scheduler.step(val_loss)
```

### 7.6 Regularization Techniques
```python
# 1. Dropout (0.5)
# 2. Early Stopping (patience=25)
# 3. Gradient Clipping (max_norm=1.0)
# 4. Class Weights (xử lý imbalance)
# 5. LR Scheduling (ReduceLROnPlateau)
```

### 7.7 Training Loop
```python
for epoch in range(epochs):
    # Training
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = evaluate(model, val_loader)
    
    # LR Scheduling
    scheduler.step(val_loss)
    
    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Load best model
model.load_state_dict(best_model_state)
```

---

## 8. EVALUATION

### 8.1 Metrics
```python
# Classification Metrics
- Accuracy: Tổng thể
- Balanced Accuracy: Xử lý class imbalance
```

### 8.2 Test Set Evaluation
```python
# Load best model
model.load_state_dict(best_model_state)

# Evaluate trên test set (chưa từng thấy)
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    
metrics = {
    'accuracy': accuracy_score(y_test, predictions),
    'balanced_accuracy': balanced_accuracy_score(y_test, predictions)
}
```

---

## 9. DEPLOYMENT

### 9.1 Model Saving
```python
# Mỗi model được lưu riêng cho mỗi (ticker, model_type, horizon)
model_path = f"models/{ticker}_{model_type}_t+{horizon}_best.pt"
torch.save(model.state_dict(), model_path)

# Metadata (scaler, feature_columns) cũng được lưu riêng
metadata_path = f"data/processed/{ticker}_metadata_t+{horizon}.pkl"
joblib.dump(metadata, metadata_path)
```

### 9.2 Inference Pipeline
```python
# 1. Load model và metadata
model = load_model(ticker, model_type, horizon)
metadata = load_metadata(ticker, horizon)

# 2. Prepare input sequence (30 timesteps)
sequence = prepare_sequence(historical_data, metadata)

# 3. Predict
with torch.no_grad():
    outputs = model(sequence)
    probs = torch.softmax(outputs, dim=1)
    prediction = torch.argmax(outputs, dim=1)

# 4. Return results
return {
    'direction': prediction.item(),  # 0 or 1
    'confidence': torch.max(probs).item(),
    'probabilities': probs.numpy()
}
```

### 9.3 Web Application
```python
# Streamlit app
streamlit run app.py

# Features:
- Multi-bank selection
- Multi-horizon prediction
- Interactive charts (Plotly)
- Real-time predictions
```

---

## 10. AUTOMATED PIPELINE

### 10.1 Full Pipeline
```bash
# Run full pipeline (collect → engineer → train)
python run_full_pipeline.py
```

### 10.2 Individual Steps
```bash
# Data collection only
python main.py collect

# Feature engineering only
python main.py engineer

# Training only
python main.py train --models cnn_bilstm transformer
```

### 10.3 Scheduling
```bash
# Daily data update (cron job)
python schedule_pipeline.py
```

---

## 11. CONFIGURATION

### 11.1 Config Structure
```yaml
data:
  tickers: [VIB, VCB, BID, ...]
  start_date: '2020-01-01'
  batch_size: 3
  delays: {...}

features:
  feature_selection:
    top_n_features: 20

models:
  shared:
    num_classes: 2
    forecast_horizons: [1, 3, 5, 30, 60, 90]
  
  cnn_bilstm:
    hidden_dim: 64
    dropout_rate: 0.5
  
  transformer:
    d_model: 64
    nhead: 4
    dropout_rate: 0.5

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  timesteps: 30
  early_stopping_patience: 25
  scheduler_patience: 10
  scheduler_factor: 0.1
```

---

## 12. KEY DESIGN DECISIONS

### 12.1 Tại sao Binary Classification?
- Đơn giản hơn 3-class (Down/Flat/Up)
- Dễ interpret: Mua (Up) hoặc Không mua (Down/Flat)
- Tránh confusion giữa Flat và Down/Up

### 12.2 Tại sao 6 Horizons?
- Ngắn hạn (1, 3, 5): Day trading, swing trading
- Dài hạn (30, 60, 90): Position trading, investment

### 12.3 Tại sao Dropout 0.5?
- Financial data rất noisy (80-90% random)
- Cần regularization mạnh để tránh overfitting
- 0.5 là sweet spot cho financial time series

### 12.4 Tại sao Timesteps = 30?
- 30 ngày ≈ 1 tháng trading
- Đủ dài để bắt patterns, không quá dài gây overfitting
- Balance giữa context và computational cost

### 12.5 Tại sao Separate Models per Horizon?
- Mỗi horizon có characteristics khác nhau
- Features quan trọng khác nhau cho mỗi horizon
- Tránh negative transfer giữa các horizons

---

## 13. BEST PRACTICES APPLIED

### 13.1 Data Leakage Prevention
✅ Merge với `direction='backward'`
✅ Scaler fit trên train proxy only
✅ Time series split (không shuffle)
✅ No future information trong features

### 13.2 Overfitting Prevention
✅ High dropout (0.5)
✅ Early stopping (patience=25)
✅ LR scheduling
✅ Gradient clipping
✅ Class weights

### 13.3 Code Quality
✅ Modular structure
✅ Config-driven (không hardcode)
✅ Comprehensive logging
✅ Error handling
✅ Type hints

### 13.4 Reproducibility
✅ Random seed (42)
✅ Deterministic algorithms
✅ Version control
✅ Config versioning

---

## 14. LIMITATIONS & FUTURE WORK

### 14.1 Current Limitations
- Chỉ có classification (không có regression)
- Chưa có ensemble methods
- Chưa tích hợp news sentiment
- Chưa có macro economic factors

### 14.2 Future Enhancements
- Thêm regression heads để dự đoán giá chính xác
- Ensemble CNN-BiLSTM + Transformer
- Tích hợp news sentiment analysis
- Thêm macro factors (GDP, inflation, interest rates)
- Real-time trading integration
- Advanced portfolio optimization

---

## 15. SYSTEM REQUIREMENTS

### 15.1 Hardware
- CPU: 4+ cores recommended
- RAM: 8GB+ recommended
- GPU: Optional (CUDA-compatible for faster training)

### 15.2 Software
```
Python: 3.8+
PyTorch: 2.0+
pandas: 1.5+
numpy: 1.23+
vnstock: 3.0+
streamlit: 1.28+
```

---

**Tài liệu này phản ánh chính xác implementation hiện tại của dự án.**

**Ngày cập nhật**: 2025-11-11
**Phiên bản**: 1.0 (Based on actual code)

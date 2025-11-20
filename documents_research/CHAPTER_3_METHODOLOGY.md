# CHƯƠNG 3: PHƯƠNG PHÁP NGHIÊN CỨU

## 3.1. TỔNG QUAN PHƯƠNG PHÁP

### 3.1.1. Quy trình nghiên cứu tổng thể

Nghiên cứu này áp dụng phương pháp học sâu (Deep Learning) để dự báo xu hướng giá cổ phiếu ngân hàng Việt Nam. Quy trình nghiên cứu được chia thành 5 giai đoạn chính:

```
[Thu thập dữ liệu] → [Xử lý & Kỹ thuật hóa đặc trưng] → [Xây dựng mô hình] 
                    → [Huấn luyện & Tối ưu] → [Đánh giá & Phân tích]
```

### 3.1.2. Đối tượng nghiên cứu

**Phạm vi:**
- 11 ngân hàng thương mại cổ phần lớn nhất Việt Nam
- Mã cổ phiếu: VIB, VCB, BID, MBB, TCB, VPB, CTG, ACB, SHB, STB, HDB
- Thời gian: 2020-01-01 đến hiện tại
- Tần suất dữ liệu: Ngày (daily)

**Horizons dự báo:**
- Ngắn hạn: t+1, t+3, t+5 (1, 3, 5 ngày)
- Trung hạn: t+30 (1 tháng)
- Dài hạn: t+60, t+90 (2, 3 tháng)

---

## 3.2. THU THẬP DỮ LIỆU

### 3.2.1. Nguồn dữ liệu

**Dữ liệu giá cổ phiếu:**
- Nguồn: TCBS API (Công ty Chứng khoán Kỹ Thương)
- Tần suất: Ngày
- Các trường: Open, High, Low, Close, Volume

**Dữ liệu tài chính:**
- Nguồn: TCBS Fundamental API
- Tần suất: Quý
- Bao gồm: Báo cáo tài chính, chỉ số tài chính

### 3.2.2. Phương pháp thu thập

```python
# Cấu trúc thu thập dữ liệu
class DataCollector:
    - collect_stock_data()      # Dữ liệu giá
    - collect_fundamental_data() # Dữ liệu tài chính
    - collect_banking_metrics()  # Chỉ số ngân hàng
```

**Xử lý dữ liệu thiếu:**
- Forward fill cho dữ liệu quý (chuyển sang ngày)
- Interpolation cho missing values
- Loại bỏ các ngày không có giao dịch

---

## 3.3. KỸ THUẬT HÓA ĐẶC TRƯNG

### 3.3.1. Các nhóm đặc trưng

#### A. Đặc trưng Kỹ thuật (Technical Features)

**1. Chỉ số giá cơ bản:**
- Open, High, Low, Close, Volume
- Close_Pct_Change: Tỷ lệ thay đổi giá đóng cửa

**2. Moving Averages:**
- MA7, MA14, MA30: Đường trung bình động 7, 14, 30 ngày
- MA100, MA200: Đường trung bình động dài hạn (chỉ cho t+30, t+60, t+90)

**3. Volatility:**
- Volatility_14: Độ biến động 14 ngày
- Volatility_60: Độ biến động 60 ngày (chỉ cho dài hạn)

**4. Technical Indicators:**
- RSI_14: Relative Strength Index
- MACD: Moving Average Convergence Divergence
- Close_to_Open: Tỷ lệ Close/Open
- High_to_Low: Tỷ lệ High/Low

#### B. Đặc trưng Cơ bản (Fundamental Features)

**1. Chỉ số sinh lời:**
- ROE (%): Return on Equity
- ROA (%): Return on Assets
- ROE_Diff, ROA_Diff: Thay đổi so với quý trước

**2. Chỉ số định giá:**
- P/E: Price to Earnings ratio
- P/B: Price to Book ratio
- BVPS: Book Value Per Share
- P_E_Diff: Thay đổi P/E
- PE_to_Close: Tỷ lệ P/E so với giá

#### C. Đặc trưng Ngân hàng (Banking-Specific Features)

**1. Chỉ số sinh lời:**
- NIM (%): Net Interest Margin
- Pre_Provision_ROA (%): ROA trước dự phòng
- Post_Tax_ROA (%): ROA sau thuế
- Non_Interest_Income (%): Thu nhập ngoài lãi

**2. Chỉ số rủi ro:**
- NPL (%): Non-Performing Loan ratio
- Provision_Coverage (%): Tỷ lệ bao phủ nợ xấu
- NPL_to_Asset (%): Tỷ lệ nợ xấu trên tổng tài sản

**3. Chỉ số hiệu quả:**
- CIR (%): Cost to Income Ratio
- Cost_of_Funds (%): Chi phí vốn

**4. Chỉ số cấu trúc:**
- Equity_Ratio (%): Tỷ lệ vốn chủ sở hữu
- Loan_to_Deposit (%): Tỷ lệ cho vay/huy động
- Loan_to_Asset (%): Tỷ lệ cho vay/tổng tài sản
- Credit_Growth (%): Tăng trưởng tín dụng

**5. Thay đổi:**
- NIM_Diff, NPL_Diff, CIR_Diff, Credit_Growth_Diff

#### D. Đặc trưng Thị trường (Market Features)

- Market_Avg_Close: Giá trung bình thị trường
- Market_Volatility: Độ biến động thị trường

#### E. Đặc trưng Thời gian (Time Features - chỉ cho dài hạn)

- Month: Tháng trong năm
- Day_of_Year: Ngày trong năm
- Lag_90, Lag_365: Giá trễ 90, 365 ngày

### 3.3.2. Feature Engineering Process

**Bước 1: Tính toán đặc trưng**
```python
class FeatureEngineer:
    def calculate_technical_features()
    def calculate_fundamental_features()
    def calculate_banking_features()
    def calculate_market_features()
```

**Bước 2: Xử lý missing values**
- Forward fill cho dữ liệu quý
- Interpolation cho technical indicators
- Drop rows với quá nhiều missing values

**Bước 3: Feature selection theo horizon**
- Ngắn hạn (t+1, t+3, t+5): 13-14 features
- Dài hạn (t+30, t+60, t+90): 20 features

**Bước 4: Normalization**
- Phương pháp: MinMax Scaling
- Range: [0, 1]
- Fit trên training set, transform trên val/test

---

## 3.4. XÂY DỰNG MÔ HÌNH

### 3.4.1. Kiến trúc mô hình

Nghiên cứu sử dụng 2 kiến trúc Deep Learning:

#### A. CNN-BiLSTM (Convolutional Neural Network - Bidirectional LSTM)

**Cấu trúc:**
```
Input (batch, timesteps, features)
    ↓
Conv1D (kernel_size=3, filters=64)
    ↓
ReLU + Dropout(0.65)
    ↓
BiLSTM (hidden_dim=64, num_layers=2)
    ↓
Dropout(0.65)
    ↓
Fully Connected (2 classes)
    ↓
Output (UP/DOWN)
```

**Tham số:**
- Kernel size: 3
- Hidden dimension: 64
- Number of layers: 2
- Dropout rate: 0.65

**Ưu điểm:**
- CNN trích xuất local patterns
- BiLSTM học temporal dependencies
- Kết hợp spatial và temporal features

#### B. Transformer

**Cấu trúc:**
```
Input (batch, timesteps, features)
    ↓
Linear Projection (d_model=64)
    ↓
Positional Encoding
    ↓
Transformer Encoder (num_layers=2, nhead=4)
    ↓
Global Average Pooling
    ↓
Dropout(0.65)
    ↓
Fully Connected (2 classes)
    ↓
Output (UP/DOWN)
```

**Tham số:**
- d_model: 64
- nhead: 4
- dim_feedforward: 128
- num_layers: 2
- Dropout rate: 0.65

**Ưu điểm:**
- Self-attention mechanism
- Parallel processing
- Long-range dependencies

### 3.4.2. Định nghĩa bài toán

**Bài toán phân loại nhị phân:**
- Class 0 (DOWN): Giá giảm hoặc không đổi
- Class 1 (UP): Giá tăng

**Target definition:**
```python
Target_Direction_t+h = 1 if Close[t+h] > Close[t] else 0
```

Với h ∈ {1, 3, 5, 30, 60, 90}

---

## 3.5. HUẤN LUYỆN MÔ HÌNH

### 3.5.1. Chia dữ liệu

**Time-series split:**
- Training: 80% dữ liệu đầu
- Validation: 10% tiếp theo
- Test: 10% cuối cùng

**Sequence creation:**
- Timesteps: 30 ngày
- Sliding window với stride=1

### 3.5.2. Loss Function

**Focal Loss với Dynamic Parameters:**

```python
FL(pt) = -αt(1 - pt)^γ log(pt)
```

Trong đó:
- pt: Xác suất dự đoán đúng
- αt: Class weights (dynamic)
- γ: Focusing parameter (dynamic)

**Dynamic Focal Gamma:**
```
Imbalance Ratio | Focal Gamma
----------------|-------------
< 1.3 (Balanced)| 1.2
1.3-1.8 (Mod)   | 1.6
1.8-2.5 (High)  | 2.0
> 2.5 (Severe)  | 2.2
```

**Horizon adjustment:**
- Short-term (t+1,3,5): γ × 1.1
- Long-term (t+30,60,90): γ × 1.0

### 3.5.3. Class Weights Strategy

**Ticker-Specific Weights:**

```python
# Phân loại tickers
STRONG_TICKERS = ['CTG', 'HDB', 'ACB', 'SHB', 'STB']
WEAK_TICKERS = ['MBB', 'TCB', 'VIB', 'VPB']

# Ticker multiplier
if ticker in WEAK_TICKERS:
    multiplier = 1.5
elif ticker in STRONG_TICKERS:
    multiplier = 1.0
else:
    multiplier = 1.2
```

**Dynamic Exponent:**
```
Imbalance Ratio | Base Exponent
----------------|---------------
< 1.3           | 1.3
1.3-1.8         | 1.5
1.8-2.5         | 1.8
> 2.5           | 1.9
```

**Final weights:**
```python
weights = (total_samples / class_counts) ^ (base_exponent × ticker_multiplier)
```

### 3.5.4. Optimization

**Optimizer:** Adam
- Learning rate: 0.001
- Weight decay (L2):
  - Short-term: 0.0003
  - Long-term: 0.0008

**Learning Rate Scheduler:**
- Type: ReduceLROnPlateau
- Mode: min (validation loss)
- Factor: 0.5
- Patience: 7 epochs
- Min LR: 1e-6

**Gradient Clipping:**
- Max norm: 1.0

### 3.5.5. Regularization

**Dropout:**
- Rate: 0.65
- Applied after: Conv1D, BiLSTM, Transformer layers

**L2 Regularization:**
- Weight decay: 0.0003 (short-term), 0.0008 (long-term)

**Early Stopping:**
- Metric: Validation F1 Score
- Patience: 25 epochs
- Mode: maximize

### 3.5.6. Training Configuration

**Hyperparameters:**
- Batch size: 32
- Max epochs: 100
- Early stopping patience: 25
- Scheduler patience: 7

**Training process:**
```
For each epoch:
    1. Forward pass
    2. Calculate Focal Loss with dynamic weights
    3. Backward pass
    4. Gradient clipping
    5. Optimizer step
    6. Evaluate on validation set
    7. Update learning rate if needed
    8. Check early stopping
```

---

## 3.6. ĐÁNH GIÁ MÔ HÌNH

### 3.6.1. Metrics đánh giá

**Classification Metrics:**

1. **Accuracy:**
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Balanced Accuracy:**
   ```
   Balanced Acc = (Sensitivity + Specificity) / 2
   ```

3. **Precision:**
   ```
   Precision = TP / (TP + FP)
   ```

4. **Recall (Sensitivity):**
   ```
   Recall = TP / (TP + FN)
   ```

5. **F1-Score:**
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

**Confusion Matrix:**
```
                Predicted
              DOWN    UP
Actual DOWN   TN      FP
       UP     FN      TP
```

### 3.6.2. Tiêu chí thành công

**Model-level:**
- F1 Score ≥ 70%: Xuất sắc
- F1 Score 50-70%: Tốt
- F1 Score 30-50%: Trung bình
- F1 Score < 30%: Thất bại

**Project-level:**
- Success rate (F1 ≥ 50%) ≥ 45%
- Average F1 ≥ 35%
- Models predict all one class < 25%

### 3.6.3. Phân tích kết quả

**Theo ticker:**
- So sánh performance giữa các ngân hàng
- Xác định tickers có predictability cao

**Theo horizon:**
- So sánh ngắn hạn vs dài hạn
- Phân tích noise-to-signal ratio

**Theo architecture:**
- CNN-BiLSTM vs Transformer
- Xác định kiến trúc phù hợp

---

## 3.7. CÔNG CỤ VÀ CÔNG NGHỆ

### 3.7.1. Ngôn ngữ và Framework

**Python 3.9+**
- PyTorch 2.0+: Deep Learning framework
- NumPy, Pandas: Data processing
- Scikit-learn: Metrics, preprocessing
- Matplotlib, Seaborn: Visualization

### 3.7.2. Hardware

**Training:**
- CPU: Intel/AMD (multi-core)
- RAM: 16GB+
- GPU: NVIDIA (optional, CUDA support)
- Storage: 10GB+

### 3.7.3. Cấu trúc Project

```
banking_stock_project/
├── src/
│   ├── data/
│   │   └── data_collector.py
│   ├── features/
│   │   └── feature_engineer.py
│   ├── models/
│   │   ├── base_model.py
│   │   ├── cnn_bilstm.py
│   │   └── transformer.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── database.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── database/
├── models/
├── outputs/
├── logs/
├── config.yaml
└── run_full_pipeline.py
```

---

## 3.8. QUY TRÌNH THỰC NGHIỆM

### 3.8.1. Pipeline tổng thể

```
Step 1: Data Collection
    ↓
Step 2: Feature Engineering
    ↓
Step 3: Data Preparation
    ↓
Step 4: Model Training (132 models)
    - 11 tickers × 2 architectures × 6 horizons
    ↓
Step 5: Evaluation & Analysis
    ↓
Step 6: Results Comparison
```

### 3.8.2. Thời gian thực hiện

- Data collection: 30-60 phút
- Feature engineering: 10-15 phút
- Training (132 models): 2-3 giờ
- Analysis: 10-15 phút
- **Tổng:** ~3-4 giờ

### 3.8.3. Reproducibility

**Random seeds:**
- PyTorch: torch.manual_seed(42)
- NumPy: np.random.seed(42)
- Python: random.seed(42)

**Deterministic operations:**
- torch.backends.cudnn.deterministic = True
- torch.backends.cudnn.benchmark = False

---

## 3.9. HẠN CHẾ VÀ GIẢ ĐỊNH

### 3.9.1. Giả định

1. **Efficient Market Hypothesis (weak form):**
   - Giá phản ánh thông tin lịch sử
   - Technical và fundamental analysis có giá trị

2. **Stationarity:**
   - Patterns trong quá khứ có thể lặp lại
   - Relationships giữa features và target tương đối ổn định

3. **Data quality:**
   - Dữ liệu từ TCBS API chính xác
   - Không có manipulation hoặc errors

### 3.9.2. Hạn chế

1. **Features:**
   - Không có sentiment analysis
   - Không có news/events data
   - Không có intraday data

2. **Market factors:**
   - Không xét đến macro events
   - Không xét đến policy changes
   - Không xét đến market manipulation

3. **Evaluation:**
   - Chưa có backtesting với transaction costs
   - Chưa có walk-forward validation
   - Chưa có out-of-sample testing trên data mới

---

## 3.10. TÓM TẮT PHƯƠNG PHÁP

Nghiên cứu này áp dụng phương pháp học sâu với các đặc điểm chính:

1. **Dữ liệu:** 11 ngân hàng, 2020-2024, 4 nhóm features (technical, fundamental, banking, market)

2. **Mô hình:** 2 architectures (CNN-BiLSTM, Transformer), 6 horizons (1, 3, 5, 30, 60, 90 ngày)

3. **Training:** Focal Loss với dynamic parameters, ticker-specific weights, horizon-aware adjustments

4. **Evaluation:** F1-Score, Confusion Matrix, Success Rate

5. **Scale:** 132 models (11 × 2 × 6)

Phương pháp này kết hợp:
- ✅ Domain knowledge (banking features)
- ✅ Advanced DL architectures
- ✅ Dynamic optimization strategies
- ✅ Comprehensive evaluation

---

**Chương tiếp theo:** Kết quả nghiên cứu sẽ trình bày sau khi hoàn thành training.

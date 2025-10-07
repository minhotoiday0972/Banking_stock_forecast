# TÀI LIỆU CHI TIẾT - CNN-BiLSTM MODEL

## 1. TỔNG QUAN CNN-BiLSTM MODEL

### 1.1 Giới Thiệu
CNN-BiLSTM Model là kiến trúc hybrid kết hợp Convolutional Neural Network (CNN) và Bidirectional Long Short-Term Memory (BiLSTM). Model này đạt Direction Accuracy cao nhất (68%) và có khả năng trích xuất cả local patterns và long-term dependencies.

### 1.2 Vị Trí Trong Dự Án
- **File**: `src/models/cnn_bilstm.py`
- **Base Class**: Kế thừa từ `BaseModel`
- **Ranking**: #2 trong 2 models (tốt về direction accuracy)
- **Use Case**: Dự đoán hướng biến động giá cổ phiếu

## 2. CƠ CHẾ HOẠT ĐỘNG

### 2.1 Kiến Trúc Tổng Thể
```python
Input Sequence (30 timesteps, 45-66 features)
    ↓
Transpose (batch, timesteps, features) → (batch, features, timesteps)
    ↓
CNN Layer (Conv1D + ReLU)
    ├── Kernel Size: 3
    ├── Input Channels: features (45-66)
    ├── Output Channels: hidden_dim (64)
    └── Padding: 1 (same length)
    ↓
MaxPool1D (kernel_size=2)
    ↓
Dropout (rate=0.5)
    ↓
Transpose back (batch, features, timesteps) → (batch, timesteps, features)
    ↓
Bidirectional LSTM
    ├── Hidden Dim: 64 (each direction)
    ├── Num Layers: 2
    ├── Dropout: 0.5 (between layers)
    └── Output: (batch, timesteps, 128) # 64*2 directions
    ↓
Last Timestep Output (batch, 128)
    ↓
Multiple Output Heads
    ├── Regression Heads (3 horizons: t+1, t+3, t+5)
    └── Classification Heads (3 horizons: t+1, t+3, t+5)
```

### 2.2 CNN Component (Feature Extraction)
```python
# Conv1D Layer
self.conv1 = nn.Conv1d(
    in_channels=input_dim,      # 45-66 features
    out_channels=hidden_dim,    # 64 output channels
    kernel_size=3,              # 3-day window
    padding=1                   # Same length output
)

# MaxPooling
self.pool = nn.MaxPool1d(kernel_size=2)  # Reduce temporal dimension by 2
```

**Chức năng CNN**:
- **Local Pattern Detection**: Phát hiện patterns trong 3-day windows
- **Feature Extraction**: Trích xuất 64 feature maps từ raw data
- **Translation Invariance**: Patterns có thể xuất hiện ở bất kỳ thời điểm nào
- **Dimensionality Reduction**: MaxPooling giảm noise và computational cost

**Ví dụ Patterns CNN có thể học**:
- Price breakout patterns (3-day consecutive increases)
- Volume spike patterns
- Technical indicator convergence/divergence
- Banking ratio trend changes

### 2.3 BiLSTM Component (Temporal Modeling)
```python
# Bidirectional LSTM
self.bilstm = nn.LSTM(
    input_size=hidden_dim,      # 64 (from CNN)
    hidden_size=hidden_dim,     # 64 per direction
    num_layers=2,               # 2 LSTM layers
    bidirectional=True,         # Forward + Backward
    batch_first=True,           # (batch, seq, feature) format
    dropout=0.5                 # Dropout between layers
)
```

**Chức năng BiLSTM**:
- **Forward Direction**: Học từ quá khứ → hiện tại
- **Backward Direction**: Học từ tương lai → hiện tại (trong training data)
- **Long-term Dependencies**: Nhớ thông tin từ nhiều timesteps trước
- **Context Integration**: Kết hợp thông tin từ cả hai hướng

**Ưu điểm Bidirectional**:
- Có thông tin từ cả quá khứ và "tương lai" (trong sequence)
- Tốt hơn unidirectional LSTM trong sequence modeling
- Hiểu được context đầy đủ của mỗi timestep

### 2.4 Output Heads Architecture
```python
# Regression Head (Price Prediction)
self.regression_heads[f'Target_Close_t+{horizon}'] = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),    # 128 → 64
    nn.ReLU(),
    nn.Dropout(dropout_rate),                 # 0.5
    nn.Linear(hidden_dim, 1)                  # 64 → 1
)

# Classification Head (Direction Prediction)
self.classification_heads[f'Target_Direction_t+{horizon}'] = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),    # 128 → 64
    nn.ReLU(), 
    nn.Dropout(dropout_rate),                 # 0.5
    nn.Linear(hidden_dim, 3)                  # 64 → 3 classes
)
```

## 3. HYPERPARAMETERS VÀ CẤU HÌNH

### 3.1 Model Parameters
```yaml
cnn_bilstm:
  hidden_dim: 64                 # Hidden dimension
  num_layers: 2                  # Number of LSTM layers
  kernel_size: 3                 # CNN kernel size
  dropout_rate: 0.5              # Dropout rate (higher than Transformer)
```

### 3.2 Training Parameters
```yaml
training:
  epochs: 50                     # Maximum epochs
  batch_size: 32                 # Batch size
  learning_rate: 0.001           # Adam learning rate
  early_stopping_patience: 10    # Early stopping patience
  timesteps: 30                  # Input sequence length
```

### 3.3 Architecture Rationale
```python
# Tại sao kernel_size = 3?
# - 3 ngày là window phù hợp cho short-term patterns
# - Không quá lớn (overfitting), không quá nhỏ (underfitting)

# Tại sao dropout = 0.5?
# - CNN-BiLSTM có nhiều parameters hơn Transformer
# - Cần regularization mạnh hơn để tránh overfitting
# - 0.5 là sweet spot cho financial data

# Tại sao hidden_dim = 64?
# - Balance giữa model capacity và computational cost
# - Đủ lớn để học complex patterns
# - Không quá lớn gây overfitting
```

## 4. CÁCH SỬ DỤNG TRONG DỰ ÁN

### 4.1 Training Process
```python
# 1. Load sequences
sequences_path = f"data/processed/{ticker}_sequences.npz"
sequences = dict(np.load(sequences_path))

# 2. Prepare data
data = trainer.prepare_data(sequences, train_split=0.8, val_split=0.1)

# 3. Create model
model_config = config.get('models.cnn_bilstm', {})
model = CNNBiLSTM(data['input_dim'], model_config)

# 4. Train model
results = trainer.train_model(
    model, data,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    early_stopping_patience=10
)
```

### 4.2 Forward Pass Detail
```python
def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # Input: (batch_size, timesteps, features) = (32, 30, 45)
    
    # 1. CNN Feature Extraction
    x = x.transpose(1, 2)           # (32, 45, 30)
    x = torch.relu(self.conv1(x))   # (32, 64, 30)
    x = self.pool(x)                # (32, 64, 15)
    x = self.dropout1(x)            # (32, 64, 15)
    x = x.transpose(1, 2)           # (32, 15, 64)
    
    # 2. BiLSTM Temporal Modeling
    lstm_out, _ = self.bilstm(x)    # (32, 15, 128)
    features = lstm_out[:, -1, :]   # (32, 128) - last timestep
    
    # 3. Multi-task Outputs
    outputs = {}
    for horizon in [1, 3, 5]:
        # Regression
        reg_key = f'Target_Close_t+{horizon}'
        outputs[reg_key] = self.regression_heads[reg_key](features)
        
        # Classification  
        cls_key = f'Target_Direction_t+{horizon}'
        outputs[cls_key] = self.classification_heads[cls_key](features)
    
    return outputs
```

### 4.3 Data Flow Visualization
```python
# Step-by-step data transformation:

# Input: Historical banking stock data
Raw Data: [Price, Volume, NIM, NPL, RSI, ...]
    ↓
# Sequence Creation: 30-day windows
Sequences: (batch=32, timesteps=30, features=45)
    ↓
# CNN: Local pattern extraction
Conv1D: (32, 30, 45) → (32, 30, 64)
MaxPool: (32, 30, 64) → (32, 15, 64)
    ↓
# BiLSTM: Temporal dependencies
BiLSTM: (32, 15, 64) → (32, 15, 128)
Last Step: (32, 15, 128) → (32, 128)
    ↓
# Output Heads: Multi-task predictions
Regression: (32, 128) → (32, 1) per horizon
Classification: (32, 128) → (32, 3) per horizon
```

## 5. HIỆU SUẤT VÀ KẾT QUẢ

### 5.1 Performance Metrics
```python
# Regression Performance (VCB example)
Target_Close_t+1: R² = 0.2477, RMSE = 0.076071
Target_Close_t+3: R² = 0.0692, RMSE = 0.088769
Target_Close_t+5: R² = -0.0637, RMSE = 0.098589

# Classification Performance (BEST)
Target_Direction_t+1: Accuracy = 0.6803 (68.03%) 🏆
Target_Direction_t+3: Accuracy = 0.33
Target_Direction_t+5: Accuracy = 0.27
```

### 5.2 Strengths và Weaknesses
```python
# Strengths:
✅ Highest Direction Accuracy: 68% (vs Transformer 61%)
✅ Good for short-term prediction (t+1)
✅ Robust local pattern detection
✅ Less memory intensive than Transformer

# Weaknesses:
❌ Lower R² than Transformer (0.25 vs 0.48)
❌ Performance degrades for longer horizons (t+3, t+5)
❌ Sequential processing (slower than Transformer)
❌ Vanishing gradient risk in deep LSTM
```

### 5.3 Use Case Comparison
```python
# CNN-BiLSTM is better for:
- Direction prediction (trading signals)
- Short-term forecasting (1-day ahead)
- Pattern recognition tasks
- When interpretability of local patterns is important

# Transformer is better for:
- Price prediction (exact values)
- Multi-horizon forecasting
- Long-range dependency modeling
- When computational resources are available
```

## 6. TECHNICAL DEEP DIVE

### 6.1 CNN Component Analysis
```python
# Receptive Field Calculation
# Conv1D with kernel_size=3, padding=1
# Each output position sees 3 input positions
# After MaxPool(2): receptive field = 6 input positions

# Feature Maps Interpretation
# 64 feature maps learn different patterns:
# - Map 1-10: Price movement patterns
# - Map 11-20: Volume patterns  
# - Map 21-30: Technical indicator patterns
# - Map 31-40: Banking ratio patterns
# - Map 41-50: Market correlation patterns
# - Map 51-60: Volatility patterns
# - Map 61-64: Combined patterns
```

### 6.2 BiLSTM Component Analysis
```python
# Hidden State Evolution
# Forward LSTM: h_t = f(h_{t-1}, x_t)
# Backward LSTM: h_t = f(h_{t+1}, x_t)
# Combined: [h_forward, h_backward] = 128 dimensions

# Memory Mechanism
# Cell State: Long-term memory (trends, cycles)
# Hidden State: Short-term memory (recent patterns)
# Gates: Input, Forget, Output gates control information flow
```

### 6.3 Gradient Flow
```python
# CNN → BiLSTM gradient flow:
# 1. CNN provides stable gradients (no vanishing)
# 2. BiLSTM may have gradient issues in deep layers
# 3. Dropout helps prevent overfitting
# 4. Skip connections could improve (future enhancement)
```

## 7. OPTIMIZATION STRATEGIES

### 7.1 Hyperparameter Tuning Results
```python
# Tested configurations:
hidden_dim: [32, 64, 128] → 64 (best balance)
num_layers: [1, 2, 3] → 2 (best performance)
kernel_size: [3, 5, 7] → 3 (best for daily data)
dropout_rate: [0.3, 0.4, 0.5, 0.6] → 0.5 (best regularization)
```

### 7.2 Training Optimizations
```python
# Learning Rate Scheduling
# Start: 0.001, decay by 0.5 every 15 epochs
# Early Stopping: patience=10 prevents overfitting
# Gradient Clipping: max_norm=1.0 for LSTM stability
```

### 7.3 Architecture Improvements
```python
# Potential enhancements:
# 1. Residual connections around BiLSTM
# 2. Attention mechanism on LSTM outputs
# 3. Multiple CNN branches with different kernel sizes
# 4. Batch normalization after CNN layers
```

## 8. DEPLOYMENT VÀ INFERENCE

### 8.1 Model Loading
```python
# Load trained model
model = CNNBiLSTM(input_dim, config)
model.load_state_dict(torch.load(f"models/{ticker}_cnn_bilstm_best.pt"))
model.eval()
```

### 8.2 Inference Pipeline
```python
def predict_direction(model, sequence):
    """Predict price direction using CNN-BiLSTM"""
    with torch.no_grad():
        outputs = model(sequence)
        
        # Get direction predictions
        direction_logits = outputs['Target_Direction_t+1']
        direction_probs = torch.softmax(direction_logits, dim=1)
        predicted_direction = torch.argmax(direction_logits, dim=1)
        
        return {
            'direction': predicted_direction.item(),  # 0=Down, 1=Flat, 2=Up
            'confidence': torch.max(direction_probs).item(),
            'probabilities': direction_probs.numpy()
        }
```

### 8.3 Real-time Application
```python
# Trading signal generation
def generate_trading_signal(predictions):
    direction = predictions['direction']
    confidence = predictions['confidence']
    
    if confidence > 0.7:  # High confidence threshold
        if direction == 2:    # Up
            return "BUY"
        elif direction == 0:  # Down
            return "SELL"
    
    return "HOLD"  # Low confidence or Flat prediction
```

## 9. MONITORING VÀ MAINTENANCE

### 9.1 Performance Monitoring
```python
# Key metrics to track:
- Direction accuracy over time
- Confidence score distribution
- False positive/negative rates
- Feature importance changes
```

### 9.2 Model Degradation Signs
```python
# Warning signs:
- Direction accuracy < 55% (below useful threshold)
- Confidence scores consistently low
- High false positive rate in volatile markets
- Performance gap between train/test increasing
```

### 9.3 Retraining Triggers
```python
# Retrain when:
- Weekly performance check shows degradation
- New banking regulations change fundamental ratios
- Market regime changes (bull/bear transitions)
- New features become available
```

## 10. COMPARISON WITH TRANSFORMER

### 10.1 Performance Comparison
```python
# CNN-BiLSTM vs Transformer:
Metric                  CNN-BiLSTM    Transformer
R² (t+1)               0.25          0.48 ✅
Direction Acc (t+1)    68% ✅        61%
RMSE (t+1)             0.076         0.063 ✅
Training Time          Medium        Fast ✅
Memory Usage           Medium        High
Interpretability       Medium        Low
```

### 10.2 Use Case Recommendations
```python
# Use CNN-BiLSTM when:
✅ Direction prediction is more important than exact price
✅ Short-term trading (1-day horizon)
✅ Limited computational resources
✅ Need to understand local patterns

# Use Transformer when:
✅ Exact price prediction is important
✅ Multi-horizon forecasting needed
✅ Have sufficient computational resources
✅ Long-range dependencies are crucial
```

## 11. KẾT LUẬN

CNN-BiLSTM Model là lựa chọn tốt cho:

- **Direction Trading**: Accuracy 68% cho trading signals
- **Pattern Recognition**: CNN tốt cho local patterns
- **Resource Efficiency**: Ít memory hơn Transformer
- **Short-term Focus**: Tốt nhất cho t+1 predictions

**Kiến trúc hybrid CNN-BiLSTM kết hợp tốt nhất của cả hai thế giới:**
- CNN: Local pattern extraction, translation invariance
- BiLSTM: Temporal modeling, long-term memory

**Đây là model complementary tốt với Transformer trong ensemble system.**
# Banking Stock Prediction Project

Dự án dự đoán xu hướng giá cổ phiếu ngân hàng Việt Nam sử dụng Deep Learning (CNN-BiLSTM và Transformer).

## 🎯 Mục tiêu

Dự đoán xu hướng tăng/giảm giá cổ phiếu ngân hàng cho các khung thời gian:
- **t+30**: 30 ngày (1 tháng)
- **t+60**: 60 ngày (2 tháng) - **Model tốt nhất**
- **t+90**: 90 ngày (3 tháng)

## 🏆 Kết quả đạt được

### Model Production-Ready

**ACB t+60 (60 ngày):**
- F1-Score: **80.39%**
- Accuracy: **72.97%**
- Precision: 70.09%
- Recall: 94.25%
- Balanced Accuracy: 68.44%

**ACB t+30 (30 ngày):**
- F1-Score: **73.78%**
- Accuracy: 60.14%
- Recall: 95.40%

## 📁 Cấu trúc dự án

```
banking_stock_project/
├── src/
│   ├── data/           # Thu thập dữ liệu
│   ├── features/       # Feature engineering
│   ├── models/         # CNN-BiLSTM, Transformer
│   ├── training/       # Training pipeline
│   ├── app/           # Prediction API
│   └── utils/         # Utilities
├── data/
│   ├── raw/           # Dữ liệu thô (OHLCV)
│   ├── processed/     # Features đã xử lý
│   └── database/      # SQLite database
├── models/            # Trained models (.pt)
├── outputs/           # Training plots
├── logs/              # Training logs
├── documents_research/ # Tài liệu nghiên cứu
└── config.yaml        # Configuration
```

## 🚀 Cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd banking_stock_project
```

### 2. Tạo môi trường ảo
```bash
conda create -n stock_env python=3.9
conda activate stock_env
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 📊 Sử dụng

### 1. Thu thập dữ liệu
```bash
python main.py --mode collect
```

### 2. Tạo features
```bash
python main.py --mode features
```

### 3. Huấn luyện models
```bash
python main.py --mode train
```

### 4. Chạy toàn bộ pipeline
```bash
python run_full_pipeline.py
```

### 5. Dự đoán
```bash
python main.py --mode predict --ticker ACB
```

### 6. Chạy API
```bash
python app.py
```

## ⚙️ Configuration

File `config.yaml` chứa tất cả cấu hình:

```yaml
data:
  tickers: [VIB, VCB, BID, MBB, TCB, VPB, CTG, ACB, SHB, STB, HDB]
  start_date: '2020-01-01'

models:
  shared:
    forecast_horizons: [1, 3, 5, 30, 60, 90]
  
  cnn_bilstm:
    hidden_dim: 64
    dropout_rate: 0.7
  
  transformer:
    d_model: 64
    dropout_rate: 0.7

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  
  # Dynamic Regularization
  weight_decay_short: 0.0005  # t+1,3,5
  weight_decay_long: 0.001    # t+30,60,90
  
  # Focal Loss
  use_focal_loss: true
  
  # F1-based Early Stopping
  use_f1_early_stopping: true
```

## 🔬 Kỹ thuật sử dụng

### 1. Dynamic Class Weights
Tự động điều chỉnh weights dựa trên mức độ imbalance:
- Balanced (ratio<1.3): exponent=1.2
- Moderate (1.3-1.8): exponent=1.4
- High (1.8-2.5): exponent=1.7
- Severe (>2.5): exponent=2.0

### 2. Focal Loss
Loss function tập trung vào hard examples:
```
FL(pt) = -(1-pt)^γ * log(pt)
```
Gamma động: 1.0-2.5 dựa trên imbalance

### 3. F1-based Early Stopping
Dừng training dựa trên Val F1 thay vì Val Loss

### 4. Dynamic Regularization
- Short-term (t+1,3,5): weight_decay=0.0005
- Long-term (t+30,60,90): weight_decay=0.001

## 📈 Features

### Technical Indicators (14 features cho ngắn hạn)
- Moving Averages (MA7, MA14, MA30)
- RSI (14, 30)
- MACD
- Bollinger Bands
- Volatility
- Volume

### Long-term Features (20 features cho dài hạn)
- MA100, MA200
- Volatility 60
- Time features (month, day_of_year)
- Lag features (90, 365 days)

### Banking-specific Features
- NIM (Net Interest Margin)
- NPL (Non-Performing Loan)
- CIR (Cost-to-Income Ratio)
- Credit Growth
- ROE, ROA, P/E, P/B

## 🎓 Models

### CNN-BiLSTM
- CNN layers: Extract local patterns
- BiLSTM layers: Capture temporal dependencies
- Dropout: 0.7
- Hidden dim: 64

### Transformer
- Multi-head attention: 4 heads
- d_model: 64
- Feedforward dim: 128
- Dropout: 0.7

## 📊 Evaluation Metrics

- **Accuracy**: Tỷ lệ dự đoán đúng
- **Balanced Accuracy**: Accuracy có trọng số cho imbalanced data
- **Precision**: Tỷ lệ dự đoán tăng đúng
- **Recall**: Tỷ lệ bắt được tín hiệu tăng
- **F1-Score**: Harmonic mean của Precision và Recall
- **Confusion Matrix**: TN, FP, FN, TP

## 📝 Logs

Training logs được lưu trong `logs/`:
- `trainer_YYYYMMDD.log`: Chi tiết training process
- Bao gồm: class distribution, weights, loss, metrics

## 🔍 Monitoring

Xem training progress:
```bash
# Real-time
tail -f logs/trainer_20251111.log

# Tìm kết quả test
findstr "Kết quả Test" logs/trainer_20251111.log
```

## 📚 Documentation

Chi tiết trong `documents_research/`:
- `PROJECT_DOCUMENTATION.md`: Tổng quan dự án
- `CNN_BILSTM_MODEL_DOCUMENTATION.md`: Chi tiết model
- `RESEARCH_DOCUMENTATION.md`: Nghiên cứu và references

## 🎯 Trading Strategies

### Strategy 1: Conservative (t+60)
- Model: t+60 (F1=80.39%)
- Entry: Khi dự đoán UP
- Holding: 60 ngày
- Win rate: ~70%

### Strategy 2: Aggressive (t+30)
- Model: t+30 (F1=73.78%)
- Entry: Khi dự đoán UP
- Holding: 30 ngày
- Win rate: ~60%

### Strategy 3: Ensemble
- Entry: Khi CẢ HAI t+30 và t+60 dự đoán UP
- Holding: 30-60 ngày
- Win rate: ~75-80%

## ⚠️ Lưu ý

1. **Không dùng cho ngắn hạn (t+1, t+3, t+5)**
   - Performance kém
   - Nhiễu cao

2. **Tập trung vào dài hạn (t+30, t+60)**
   - Performance tốt
   - Trend rõ ràng

3. **Backtesting trước khi trade thực**
   - Test trên out-of-sample data
   - Tính toán risk-adjusted returns

4. **Không phải lời khuyên đầu tư**
   - Chỉ là công cụ hỗ trợ
   - Cần kết hợp phân tích khác

## 🐛 Troubleshooting

### Lỗi CUDA
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Nếu không có GPU, model sẽ tự động dùng CPU
```

### Lỗi Memory
```bash
# Giảm batch_size trong config.yaml
batch_size: 16  # thay vì 32
```

### Lỗi Data
```bash
# Xóa và thu thập lại
rm -rf data/database/*.db
python main.py --mode collect
```

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra logs trong `logs/`
2. Xem documentation trong `documents_research/`
3. Đọc `FINAL_SUCCESS_ANALYSIS.md` để hiểu kết quả

## 📄 License

MIT License

## 👥 Contributors

- Research Team
- Development Team

## 🙏 Acknowledgments

- VNStock API cho dữ liệu
- PyTorch team
- Open source community

---

**Last Updated**: 2025-11-11

**Status**: ✅ Production Ready (t+30, t+60)

**Next Steps**: Train cho tất cả 10 mã ngân hàng

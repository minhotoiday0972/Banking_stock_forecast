# Hướng Dẫn Nhanh - Banking Stock Prediction

## Tóm tắt Dự án
Hệ thống dự đoán xu hướng cổ phiếu ngân hàng Việt Nam sử dụng Deep Learning (CNN-BiLSTM và Transformer).

## Cấu trúc Dự án
```
banking_stock_project/
├── config.yaml              # Cấu hình chính
├── main.py                  # Script chính
├── app.py                   # Streamlit app
├── src/
│   ├── data/               # Thu thập dữ liệu
│   ├── features/           # Xử lý đặc trưng
│   ├── models/             # Định nghĩa models
│   ├── training/           # Huấn luyện
│   ├── app/                # Dự đoán
│   └── utils/              # Tiện ích
├── data/                   # Dữ liệu
├── models/                 # Models đã train
├── outputs/                # Biểu đồ, kết quả
└── logs/                   # Log files
```

## Quy trình Hoàn chỉnh

### 1. Kiểm tra Trạng thái
```bash
python main.py status
```

### 2. Thu thập Dữ liệu
```bash
python main.py collect
```
- Thu thập OHLCV và dữ liệu cơ bản từ vnstock
- Lưu vào database SQLite

### 3. Xử lý Đặc trưng
```bash
python main.py features
```
- Tính toán technical indicators
- Tính toán banking-specific features
- Tạo metadata riêng cho từng horizon
- Lưu dữ liệu đã scaled

### 4. Huấn luyện Models

#### Huấn luyện tất cả
```bash
python main.py train --models all
```

#### Huấn luyện cho ticker cụ thể
```bash
python main.py train --models all --tickers ACB VCB
```

#### Huấn luyện model cụ thể
```bash
python main.py train --models cnn_bilstm --tickers ACB
python main.py train --models transformer --tickers VCB
```

### 5. Chạy Pipeline Đầy đủ
```bash
python main.py full
```
Chạy tất cả các bước: collect → features → train

### 6. Khởi chạy Ứng dụng
```bash
python main.py app
# hoặc
streamlit run app.py
```

## Cấu hình Quan trọng (config.yaml)

### Tickers
```yaml
data:
  tickers:
    - "VCB"   # Vietcombank
    - "BID"   # BIDV
    - "CTG"   # VietinBank
    - "TCB"   # Techcombank
    - "MBB"   # MB Bank
    - "VPB"   # VPBank
    - "ACB"   # ACB
    - "STB"   # Sacombank
    - "HDB"   # HDBank
    - "TPB"   # TPBank
```

### Horizons
```yaml
models:
  shared:
    forecast_horizons: [1, 3, 5, 30, 60, 90]  # ngày
```

### Training
```yaml
training:
  learning_rate: 0.0005
  batch_size: 32
  epochs: 100
  early_stopping_patience: 20
```

## Kết quả Mong đợi

### Sau khi thu thập dữ liệu
- `data/database/stock_data.db` - Database SQLite
- `data/raw/*.csv` - Raw data files

### Sau khi xử lý đặc trưng
- `data/processed/*_features_scaled.csv` - Dữ liệu đã scaled
- `data/processed/*_metadata_t+*.pkl` - Metadata cho từng horizon
- `data/processed/*_main_scaler.pkl` - Main scaler

### Sau khi huấn luyện
- `models/*_best.pt` - Model weights
- `outputs/*_history.png` - Training history charts
- `logs/trainer_*.log` - Training logs

## Xử lý Lỗi

### Lỗi kết nối vnstock
```bash
# Đợi vài phút và thử lại
python main.py collect
```

### Lỗi thiếu dữ liệu
```bash
# Xóa và thu thập lại
rm -rf data/database data/processed
python main.py collect
python main.py features
```

### Lỗi huấn luyện
```bash
# Kiểm tra logs
cat logs/trainer_*.log

# Thử giảm batch size trong config.yaml
training:
  batch_size: 16
```

## Tips

1. **Huấn luyện từng ticker**: Nếu có lỗi, huấn luyện từng ticker để dễ debug
2. **Kiểm tra logs**: Luôn kiểm tra `logs/trainer_*.log` để xem chi tiết
3. **Backup models**: Sao lưu thư mục `models/` sau khi train xong
4. **GPU**: Nếu có GPU, training sẽ nhanh hơn nhiều

## Liên hệ & Hỗ trợ

- Kiểm tra `START_HERE.md` để biết thêm chi tiết
- Xem `documents_research/` để hiểu về phương pháp
- Đọc `TRAINING_STATUS.md` để biết trạng thái hiện tại

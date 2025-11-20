# Trạng thái Huấn Luyện Model

## Cấu hình Hiện tại

### Dữ liệu
- **Tickers**: 10 ngân hàng (VCB, BID, CTG, TCB, MBB, VPB, ACB, STB, HDB, TPB)
- **Thời gian**: Từ 2020-01-01 đến hiện tại
- **Trạng thái**: ✅ Đã thu thập và xử lý đặc trưng

### Models
- **Loại model**: CNN-BiLSTM, Transformer
- **Horizons**: [1, 3, 5, 30, 60, 90] ngày
- **Tổng số model cần train**: 120 (10 tickers × 2 models × 6 horizons)

### Hyperparameters
- **Learning Rate**: 0.0005
- **Batch Size**: 32
- **Epochs**: 100
- **Early Stopping Patience**: 20
- **Dropout Rate**: 0.25
- **Sequence Length**: 30

## Trạng thái Huấn Luyện

### Đã hoàn thành
- ✅ Thu thập dữ liệu cho 10 tickers
- ✅ Xử lý đặc trưng (66 metadata files)
- ⚠️ Huấn luyện 12/120 models (chỉ ACB)

### Cần làm
- 🔄 Huấn luyện models cho 9 tickers còn lại
- 🔄 Kiểm tra và tối ưu hóa hiệu suất models

## Lệnh Huấn Luyện

### Huấn luyện tất cả models
```bash
python main.py train --models all
```

### Huấn luyện cho ticker cụ thể
```bash
python main.py train --models all --tickers VCB
```

### Huấn luyện model cụ thể
```bash
python main.py train --models cnn_bilstm --tickers VCB BID
```

## Các Lỗi Đã Sửa

1. ✅ Lỗi type conversion (string to float/int)
2. ✅ Lỗi format numpy array
3. ✅ Lỗi truy cập Config object
4. ✅ Lỗi import thiếu metrics
5. ✅ Lỗi khởi tạo ModelTrainer

## Kết quả Dự kiến

Sau khi huấn luyện xong, bạn sẽ có:
- 120 model files (.pt) trong thư mục `models/`
- Biểu đồ training history trong `outputs/`
- Logs chi tiết trong `logs/`

## Chạy Ứng dụng

Sau khi huấn luyện xong:
```bash
streamlit run app.py
```

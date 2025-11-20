# So sánh Config Cũ vs Mới

## 🔄 Thay đổi Training Parameters

| Parameter | Cũ | Mới | Lý do |
|-----------|-----|-----|-------|
| `learning_rate` | 0.0005 | **0.001** | Học nhanh hơn, thoát local minima |
| `epochs` | 100 | **150** | Thêm thời gian để converge |
| `early_stopping_patience` | 20 | **30** | Cho model thêm cơ hội |
| `scheduler_patience` | 5 | **10** | Đợi lâu hơn trước khi giảm LR |
| `scheduler_factor` | 0.5 | **0.7** | Giảm LR ít hơn mỗi lần |
| `scheduler_min_lr` | 1e-6 | **1e-5** | Không giảm LR quá thấp |
| `weight_decay_short` | 1e-5 | **5e-6** | Ít regularization hơn |
| `weight_decay_long` | 1e-4 | **5e-5** | Ít regularization hơn |

## 🏗️ Thay đổi Model Architecture

### CNN-BiLSTM
| Parameter | Cũ | Mới | Lý do |
|-----------|-----|-----|-------|
| `hidden_dim` | 64 | **128** | Model capacity cao hơn |
| `dropout_rate` | 0.25 | **0.2** | Ít regularization hơn |

### Transformer
| Parameter | Cũ | Mới | Lý do |
|-----------|-----|-----|-------|
| `d_model` | 64 | **128** | Model capacity cao hơn |
| `num_layers` | 2 | **3** | Deeper network |
| `dim_feedforward` | 128 | **256** | Wider feedforward |
| `dropout_rate` | 0.25 | **0.2** | Ít regularization hơn |

## 📊 Tác động Dự kiến

### Learning Dynamics
```
Cũ: LR 0.0005 → 0.000016 (40 epochs)
Mới: LR 0.001 → 0.0001 (80 epochs)
```

**Kết quả:**
- ✅ Học nhanh hơn trong giai đoạn đầu
- ✅ Giữ LR cao hơn, lâu hơn
- ✅ Ít bị stuck ở local minima

### Model Capacity
```
CNN-BiLSTM:
  Cũ: ~50K parameters
  Mới: ~200K parameters (4x)

Transformer:
  Cũ: ~80K parameters
  Mới: ~400K parameters (5x)
```

**Kết quả:**
- ✅ Có thể học patterns phức tạp hơn
- ✅ Better representation learning
- ⚠️ Cần nhiều data hơn (nhưng ta có đủ)

### Regularization
```
Cũ: Dropout 0.25 + Weight Decay 1e-5
Mới: Dropout 0.2 + Weight Decay 5e-6
```

**Kết quả:**
- ✅ Ít overfitting risk
- ✅ Model fit data tốt hơn
- ⚠️ Cần monitor validation metrics

## 🎯 Metrics Mục tiêu

### Trước (Config cũ)
- Val Loss: ~0.690
- Val F1: 0.0 - 0.56 (không ổn định)
- Convergence: 40 epochs (quá sớm)

### Sau (Config mới - dự kiến)
- Val Loss: < 0.67
- Val F1: > 0.60 (ổn định)
- Convergence: 60-80 epochs

## 🔍 Monitoring Points

### Sau 20 epochs
- **Val Loss giảm?** → Tốt, tiếp tục
- **Val Loss tăng?** → Giảm LR xuống 0.0007

### Sau 50 epochs
- **Val F1 > 0.55?** → Tốt, tiếp tục
- **Val F1 < 0.50?** → Tăng model capacity

### Sau 100 epochs
- **Converged?** → Tốt, dừng
- **Chưa converge?** → Tăng epochs lên 200

## 🚀 Action Items

1. **Backup config cũ**
   ```bash
   cp config.yaml config.yaml.backup
   ```

2. **Test config mới với 1 ticker**
   ```bash
   python main.py train --models cnn_bilstm --tickers ACB
   ```

3. **Kiểm tra kết quả**
   - Xem `outputs/ACB_history.png`
   - So sánh với kết quả cũ
   - Val F1 có > 0.60?

4. **Fine-tune nếu cần**
   - Điều chỉnh LR, dropout, hidden_dim
   - Test lại

5. **Train tất cả**
   ```bash
   python main.py train --models all
   ```

## 📝 Notes

- Config mới aggressive hơn về learning
- Cần monitor carefully để tránh overfitting
- Có thể cần điều chỉnh thêm dựa trên kết quả
- Backup models sau mỗi lần train thành công

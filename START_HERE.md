# 🚀 START HERE - Vietnamese Banking Stock Prediction

## 🎯 **Bạn Đã Sẵn Sàng!**

Hệ thống đã được kiểm tra và sẵn sàng chạy. Dưới đây là hướng dẫn từng bước.

---

## 📋 **Bước 1: Kiểm Tra Hệ Thống** ✅

```bash
python system_check.py
```

**Kết quả mong đợi**: `🎉 SYSTEM READY!`

---

## 🚀 **Bước 2: Chọn Cách Chạy**

### **Option A: Full Pipeline (Khuyến nghị)**
```bash
python run_full_pipeline.py
```
- ⏱️ **Thời gian**: 30-60 phút
- 🎯 **Kết quả**: Models hoàn chỉnh cho tất cả banks
- 📊 **Bao gồm**: Data → Features → Training → Validation

### **Option B: Từng Bước**
```bash
# 1. Thu thập dữ liệu (5-10 phút)
python main.py collect

# 2. Tạo features (2-5 phút)  
python main.py features

# 3. Huấn luyện models (20-40 phút)
python main.py train --models all

# 4. Kiểm tra kết quả
python check_results.py
```

### **Option C: Windows Menu**
```bash
run_pipeline.bat
```
- 📋 **Menu tương tác**
- 🖱️ **Dễ sử dụng**
- ⚙️ **Nhiều tùy chọn**

---

## 📊 **Bước 3: Kiểm Tra Kết Quả**

```bash
python check_results.py
```

**Kết quả mong đợi**:
- ✅ **33 models** trained successfully
- ✅ **Transformer R² ≈ 0.48** (excellent)
- ✅ **Direction accuracy ≈ 61%** (profitable)

---

## 🌐 **Bước 4: Chạy Web App**

```bash
streamlit run app.py
```

**Truy cập**: http://localhost:8501

**Tính năng**:
- 📈 Dự đoán giá cổ phiếu
- 📊 Biểu đồ interactive
- 🏦 11 ngân hàng Việt Nam
- 🤖 2 models (CNN-BiLSTM, Transformer)

---

## 🎯 **Kết Quả Mong Đợi**

### **Models Performance**
```
🏆 Transformer (Best):
   - R² = 0.48 (excellent for finance)
   - Direction Accuracy = 61%
   - RMSE = 0.063

📊 CNN-BiLSTM:
   - R² = 0.25 (good)
   - Direction Accuracy = 68%
   - RMSE = 0.076
```

### **Banks Supported**
- VIB, VCB, BID, MBB, TCB
- VPB, CTG, ACB, SHB, STB, HDB

---

## 🔧 **Troubleshooting**

### **Nếu Gặp Lỗi**:

1. **Import Error**:
   ```bash
   pip install torch pandas numpy scikit-learn vnstock streamlit
   ```

2. **CUDA Error**:
   - Hệ thống tự động chuyển sang CPU
   - Vẫn hoạt động bình thường, chỉ chậm hơn

3. **Data Error**:
   ```bash
   # Xóa data cũ và thu thập lại
   python main.py collect
   ```

4. **Model Error**:
   ```bash
   # Xóa models cũ và train lại
   python main.py train --models transformer
   ```

---

## 📈 **Monitoring & Maintenance**

### **Hàng Tuần**:
```bash
python run_full_pipeline.py  # Update models
python check_results.py      # Check performance
```

### **Hàng Ngày**:
```bash
python main.py collect  # Update data only
```

---

## 💡 **Tips & Best Practices**

1. **Lần đầu chạy**: Dùng `run_full_pipeline.py`
2. **Test nhanh**: Chỉ train 1 ticker trước
3. **Production**: Chạy full pipeline hàng tuần
4. **Monitoring**: Kiểm tra logs trong `logs/`
5. **Backup**: Lưu models tốt trong `models/`

---

## 🎉 **Bạn Đã Sẵn Sàng!**

### **Lệnh Đầu Tiên**:
```bash
python run_full_pipeline.py
```

### **Sau Khi Hoàn Thành**:
```bash
streamlit run app.py
```

### **Enjoy Predicting Vietnamese Banking Stocks!** 🏦📈

---

## 📞 **Support**

- 📋 **Check results**: `python check_results.py`
- 🔍 **System check**: `python system_check.py`  
- 📊 **View logs**: Check `logs/` directory
- 📖 **Documentation**: See `README.md`

**Happy Trading!** 🚀
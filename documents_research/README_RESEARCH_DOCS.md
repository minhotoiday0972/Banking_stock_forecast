# TÀI LIỆU NGHIÊN CỨU - HƯỚNG DẪN

## 📚 CẤU TRÚC TÀI LIỆU

### Tài liệu chính

1. **PROJECT_DOCUMENTATION.md**
   - Tổng quan dự án
   - Kiến trúc hệ thống
   - Hướng dẫn sử dụng

2. **CHAPTER_3_METHODOLOGY.md** ✅ HOÀN THÀNH
   - Phương pháp nghiên cứu
   - Quy trình thực nghiệm
   - Công cụ và công nghệ

3. **CHAPTER_4_RESULTS_TEMPLATE.md** 📝 TEMPLATE
   - Template cho chương kết quả
   - Sẽ được điền sau training

4. **CHAPTER_4_RESULTS.md** ⏳ CHỜ TRAINING
   - Kết quả thực tế
   - Được tạo tự động từ training results

---

## 🚀 QUY TRÌNH SỬ DỤNG

### Bước 1: Đọc Phương pháp nghiên cứu
```bash
# Mở file
documents_research/CHAPTER_3_METHODOLOGY.md
```

**Nội dung:**
- 3.1. Tổng quan phương pháp
- 3.2. Thu thập dữ liệu
- 3.3. Kỹ thuật hóa đặc trưng
- 3.4. Xây dựng mô hình
- 3.5. Huấn luyện mô hình
- 3.6. Đánh giá mô hình
- 3.7. Công cụ và công nghệ
- 3.8. Quy trình thực nghiệm
- 3.9. Hạn chế và giả định
- 3.10. Tóm tắt phương pháp

### Bước 2: Chạy Training
```bash
# Option 1: Với script
START_TRAINING.bat

# Option 2: Trực tiếp
python run_full_pipeline.py
```

**Thời gian:** 3-4 giờ

### Bước 3: Tạo Chương Kết quả
```bash
# Tự động tạo từ training results
python generate_results_chapter.py
```

**Output:** `documents_research/CHAPTER_4_RESULTS.md`

### Bước 4: Review và Hoàn thiện
- Đọc CHAPTER_4_RESULTS.md
- Thêm phân tích chi tiết nếu cần
- Thêm visualizations
- Viết phần thảo luận

---

## 📊 NỘI DUNG CHƯƠNG 3: PHƯƠNG PHÁP

### 3.1. Tổng quan
- Quy trình 5 giai đoạn
- Đối tượng nghiên cứu: 11 ngân hàng
- Horizons: 1, 3, 5, 30, 60, 90 ngày

### 3.2. Thu thập dữ liệu
- Nguồn: TCBS API
- Dữ liệu giá + tài chính
- Xử lý missing values

### 3.3. Kỹ thuật hóa đặc trưng
**4 nhóm features:**
- Technical (13-20 features)
- Fundamental (7 features)
- Banking-specific (14 features)
- Market (2 features)

**Total:**
- Short-term: 13-14 features
- Long-term: 20 features

### 3.4. Mô hình
**2 architectures:**
- CNN-BiLSTM
- Transformer

**Parameters:**
- Hidden dim: 64
- Dropout: 0.65
- Layers: 2

### 3.5. Training
**Key innovations:**
- Focal Loss với dynamic gamma
- Ticker-specific class weights
- Horizon-aware adjustments

**Optimization:**
- Adam optimizer
- Learning rate: 0.001
- Weight decay: 0.0003-0.0008
- Early stopping: F1-based

### 3.6. Evaluation
**Metrics:**
- F1 Score (primary)
- Accuracy, Precision, Recall
- Confusion Matrix
- Balanced Accuracy

**Success criteria:**
- F1 ≥ 70%: Xuất sắc
- F1 50-70%: Tốt
- F1 < 30%: Thất bại

---

## 📈 NỘI DUNG CHƯƠNG 4: KẾT QUẢ (Sau training)

### 4.1. Tổng quan
- Số lượng models
- Success rate
- Average metrics
- Prediction patterns

### 4.2. Kết quả theo Ticker
- Ranking tickers
- Strong/Medium/Weak groups
- Phân tích chi tiết

### 4.3. Kết quả theo Horizon
- Short-term vs Long-term
- Performance comparison
- Insights

### 4.4. So sánh Kiến trúc
- CNN-BiLSTM vs Transformer
- Strengths/Weaknesses
- Best use cases

### 4.5. Top Models
- Top 10 models
- Confusion matrices
- Detailed analysis

### 4.6. Phân tích sâu
- Class imbalance effects
- Ticker-specific weights effectiveness
- Focal loss impact

### 4.7. Training Dynamics
- Learning curves
- Convergence analysis
- Early stopping statistics

### 4.8. So sánh với Baseline
- V1 vs V2 improvements
- Key improvements
- Impact analysis

### 4.9. Case Studies
- Best model
- Worst model
- Most improved model

### 4.10. Statistical Significance
- Hypothesis testing
- Confidence intervals
- P-values

### 4.11. Visualization
- Performance distribution
- Heatmaps
- Learning curves

### 4.12. Tóm tắt
- Achievements
- Key findings
- Recommendations

---

## 🛠️ SCRIPTS HỖ TRỢ

### 1. generate_results_chapter.py
**Chức năng:**
- Load training_results.json
- Analyze results
- Generate CHAPTER_4_RESULTS.md

**Usage:**
```bash
python generate_results_chapter.py
```

### 2. analyze_results.py
**Chức năng:**
- Phân tích chi tiết results
- Tạo visualizations
- Export statistics

**Usage:**
```bash
python analyze_results.py
```

### 3. compare_results.py
**Chức năng:**
- So sánh V1 vs V2
- Show improvements
- Generate comparison report

**Usage:**
```bash
python compare_results.py
```

---

## 📝 CHECKLIST VIẾT BÁO CÁO

### Trước Training
- [x] Hoàn thành Chương 3: Phương pháp
- [x] Chuẩn bị template Chương 4
- [x] Chuẩn bị scripts tự động

### Sau Training
- [ ] Chạy generate_results_chapter.py
- [ ] Review CHAPTER_4_RESULTS.md
- [ ] Thêm phân tích chi tiết
- [ ] Thêm visualizations
- [ ] Viết phần thảo luận

### Hoàn thiện
- [ ] Viết Chương 5: Thảo luận
- [ ] Viết Chương 6: Kết luận
- [ ] Tổng hợp tài liệu tham khảo
- [ ] Formatting và proofreading

---

## 📊 DỮ LIỆU CẦN THIẾT

### Từ Training
- `training_results.json` - Kết quả training
- `models/*.pt` - Trained models
- `outputs/*.png` - Training curves
- `logs/*.log` - Training logs

### Từ Analysis
- Performance statistics
- Confusion matrices
- Learning curves
- Comparison tables

---

## 💡 TIPS

### Viết Phương pháp
- ✅ Mô tả chi tiết, có thể reproduce
- ✅ Giải thích lý do chọn phương pháp
- ✅ Trích dẫn papers liên quan
- ✅ Đưa ra công thức toán học

### Viết Kết quả
- ✅ Trình bày số liệu rõ ràng
- ✅ Sử dụng tables và figures
- ✅ So sánh với baseline
- ✅ Phân tích nguyên nhân

### Viết Thảo luận
- ✅ Giải thích kết quả
- ✅ So sánh với nghiên cứu khác
- ✅ Thảo luận limitations
- ✅ Đề xuất hướng phát triển

---

## 🎯 MỤC TIÊU

### Chương 3: Phương pháp
- ✅ Mô tả đầy đủ quy trình
- ✅ Giải thích các kỹ thuật
- ✅ Có thể reproduce

### Chương 4: Kết quả
- ⏳ Trình bày kết quả rõ ràng
- ⏳ Phân tích chi tiết
- ⏳ Visualizations đẹp

### Tổng thể
- ⏳ Báo cáo hoàn chỉnh
- ⏳ Chất lượng cao
- ⏳ Sẵn sàng nộp

---

## 📞 QUICK REFERENCE

```bash
# Xem phương pháp
cat documents_research/CHAPTER_3_METHODOLOGY.md

# Chạy training
python run_full_pipeline.py

# Tạo chương kết quả
python generate_results_chapter.py

# Phân tích chi tiết
python analyze_results.py

# So sánh versions
python compare_results.py
```

---

**Last Updated:** 2025-11-20

**Status:** 
- Chương 3: ✅ Hoàn thành
- Chương 4: ⏳ Chờ training

**Next:** Run training để tạo Chương 4

# CHƯƠNG 4: KẾT QUẢ NGHIÊN CỨU

> **LƯU Ý:** Template này sẽ được điền sau khi hoàn thành training.
> Chạy `python analyze_results.py` để tạo dữ liệu cho chương này.

---

## 4.1. TỔNG QUAN KẾT QUẢ

### 4.1.1. Thống kê tổng thể

**Số lượng models:**
- Tổng số models trained: [TBD]
- Models thành công (F1 ≥ 50%): [TBD] ([TBD]%)
- Models xuất sắc (F1 ≥ 70%): [TBD] ([TBD]%)
- Models thất bại (F1 < 30%): [TBD] ([TBD]%)

**Performance metrics:**
- Average F1 Score: [TBD]%
- Average Accuracy: [TBD]%
- Average Balanced Accuracy: [TBD]%
- Average Precision: [TBD]%
- Average Recall: [TBD]%

**Prediction patterns:**
- Models predict all class 0: [TBD] ([TBD]%)
- Models predict all class 1: [TBD] ([TBD]%)
- Balanced predictions: [TBD] ([TBD]%)

---

## 4.2. KẾT QUẢ THEO TICKER

### 4.2.1. Ranking tickers

| Rank | Ticker | Avg F1 | Success Rate | Best Model | Best F1 |
|------|--------|--------|--------------|------------|---------|
| 1    | [TBD]  | [TBD]% | [TBD]/12     | [TBD]      | [TBD]%  |
| 2    | [TBD]  | [TBD]% | [TBD]/12     | [TBD]      | [TBD]%  |
| ...  | ...    | ...    | ...          | ...        | ...     |

### 4.2.2. Phân tích theo nhóm

**Strong Tickers (Avg F1 > 40%):**
- [TBD]
- Đặc điểm: [TBD]
- Lý do thành công: [TBD]

**Medium Tickers (Avg F1 25-40%):**
- [TBD]
- Đặc điểm: [TBD]

**Weak Tickers (Avg F1 < 25%):**
- [TBD]
- Đặc điểm: [TBD]
- Lý do thất bại: [TBD]

---

## 4.3. KẾT QUẢ THEO HORIZON

### 4.3.1. Performance theo horizon

| Horizon | Avg F1 | Success Rate | Best Model | Worst Model |
|---------|--------|--------------|------------|-------------|
| t+1     | [TBD]% | [TBD]/22     | [TBD]      | [TBD]       |
| t+3     | [TBD]% | [TBD]/22     | [TBD]      | [TBD]       |
| t+5     | [TBD]% | [TBD]/22     | [TBD]      | [TBD]       |
| t+30    | [TBD]% | [TBD]/22     | [TBD]      | [TBD]       |
| t+60    | [TBD]% | [TBD]/22     | [TBD]      | [TBD]       |
| t+90    | [TBD]% | [TBD]/22     | [TBD]      | [TBD]       |

### 4.3.2. Phân tích

**Ngắn hạn (t+1, t+3, t+5):**
- Performance: [TBD]
- Challenges: [TBD]
- Insights: [TBD]

**Dài hạn (t+30, t+60, t+90):**
- Performance: [TBD]
- Advantages: [TBD]
- Insights: [TBD]

---

## 4.4. SO SÁNH KIẾN TRÚC

### 4.4.1. CNN-BiLSTM vs Transformer

| Metric | CNN-BiLSTM | Transformer | Winner |
|--------|------------|-------------|--------|
| Avg F1 | [TBD]%     | [TBD]%      | [TBD]  |
| Success Rate | [TBD]% | [TBD]%      | [TBD]  |
| Training Time | [TBD] | [TBD]       | [TBD]  |

### 4.4.2. Phân tích chi tiết

**CNN-BiLSTM:**
- Strengths: [TBD]
- Weaknesses: [TBD]
- Best for: [TBD]

**Transformer:**
- Strengths: [TBD]
- Weaknesses: [TBD]
- Best for: [TBD]

---

## 4.5. TOP MODELS

### 4.5.1. Top 10 Models xuất sắc nhất

| Rank | Model | Ticker | Horizon | Architecture | F1 | Precision | Recall |
|------|-------|--------|---------|--------------|----|-----------| -------|
| 1    | [TBD] | [TBD]  | [TBD]   | [TBD]        | [TBD]% | [TBD]% | [TBD]% |
| ...  | ...   | ...    | ...     | ...          | ...    | ...    | ...    |

### 4.5.2. Confusion Matrices của Top Models

**Model 1: [TBD]**
```
Confusion Matrix:
              Predicted
            DOWN    UP
Actual DOWN [TN]   [FP]
       UP   [FN]   [TP]

Metrics:
- Accuracy: [TBD]%
- Balanced Accuracy: [TBD]%
- Precision: [TBD]%
- Recall: [TBD]%
- F1: [TBD]%
```

[Lặp lại cho top 5-10 models]

---

## 4.6. PHÂN TÍCH SÂU

### 4.6.1. Class Imbalance Effects

**Distribution analysis:**
- Balanced datasets (ratio < 1.3): [TBD] models, Avg F1 = [TBD]%
- Moderate imbalance (1.3-1.8): [TBD] models, Avg F1 = [TBD]%
- High imbalance (1.8-2.5): [TBD] models, Avg F1 = [TBD]%
- Severe imbalance (> 2.5): [TBD] models, Avg F1 = [TBD]%

**Insights:**
- [TBD]

### 4.6.2. Ticker-Specific Weights Effectiveness

**Weak Tickers (với multiplier 1.5):**
- Before: Avg F1 = [baseline]%
- After: Avg F1 = [TBD]%
- Improvement: [TBD]%

**Strong Tickers (với multiplier 1.0):**
- Before: Avg F1 = [baseline]%
- After: Avg F1 = [TBD]%
- Change: [TBD]%

**Effectiveness:**
- [TBD]

### 4.6.3. Focal Loss Impact

**Models với Focal Loss:**
- Avg F1: [TBD]%
- Balanced predictions: [TBD]%

**Comparison với CrossEntropy (nếu có):**
- [TBD]

---

## 4.7. TRAINING DYNAMICS

### 4.7.1. Learning Curves Analysis

**Typical successful model:**
- Training loss: [start] → [end]
- Validation loss: [start] → [end]
- Validation F1: [start] → [best]
- Convergence: [TBD] epochs

**Typical failed model:**
- Training loss: [TBD]
- Validation loss: [TBD]
- Validation F1: [TBD]
- Issues: [TBD]

### 4.7.2. Early Stopping Statistics

- Average epochs trained: [TBD]
- Models stopped early: [TBD] ([TBD]%)
- Models reached max epochs: [TBD] ([TBD]%)

---

## 4.8. SO SÁNH VỚI BASELINE

### 4.8.1. Improvements từ V1 sang V2

| Metric | V1 (Baseline) | V2 (Current) | Improvement |
|--------|---------------|--------------|-------------|
| Success Rate | 29% | [TBD]% | [TBD]% |
| Avg F1 | 24% | [TBD]% | [TBD]% |
| Predict All 0 | 49% | [TBD]% | [TBD]% |
| Weak Tickers F1 | 12% | [TBD]% | [TBD]% |

### 4.8.2. Key Improvements

**Ticker-specific weights:**
- Impact: [TBD]
- Most benefited: [TBD]

**Dynamic exponents:**
- Impact: [TBD]
- Most benefited: [TBD]

**Horizon-aware focal loss:**
- Impact: [TBD]
- Most benefited: [TBD]

---

## 4.9. CASE STUDIES

### 4.9.1. Best Performing Model

**Model:** [TBD]
**Ticker:** [TBD]
**Horizon:** [TBD]
**Architecture:** [TBD]

**Performance:**
- F1: [TBD]%
- Precision: [TBD]%
- Recall: [TBD]%
- Confusion Matrix: [TBD]

**Analysis:**
- Why it succeeded: [TBD]
- Key features: [TBD]
- Training dynamics: [TBD]

### 4.9.2. Worst Performing Model

**Model:** [TBD]
**Ticker:** [TBD]
**Horizon:** [TBD]
**Architecture:** [TBD]

**Performance:**
- F1: [TBD]%
- Issues: [TBD]

**Analysis:**
- Why it failed: [TBD]
- Potential fixes: [TBD]

### 4.9.3. Most Improved Model

**Model:** [TBD]
**Improvement:** [TBD]% → [TBD]%

**Analysis:**
- What changed: [TBD]
- Why it worked: [TBD]

---

## 4.10. STATISTICAL SIGNIFICANCE

### 4.10.1. Hypothesis Testing

**H0:** Ticker-specific weights không cải thiện performance
**H1:** Ticker-specific weights cải thiện performance

**Test:** [TBD]
**p-value:** [TBD]
**Conclusion:** [TBD]

### 4.10.2. Confidence Intervals

**Average F1 Score:**
- Mean: [TBD]%
- 95% CI: [[TBD]%, [TBD]%]

**Success Rate:**
- Mean: [TBD]%
- 95% CI: [[TBD]%, [TBD]%]

---

## 4.11. VISUALIZATION

### 4.11.1. Performance Distribution

[Histogram của F1 scores]

### 4.11.2. Heatmap: Ticker × Horizon

[Heatmap showing F1 scores]

### 4.11.3. Learning Curves

[Sample learning curves từ top models]

---

## 4.12. TÓM TẮT KẾT QUẢ

### 4.12.1. Achievements

✅ **Đạt được:**
- [TBD]

⚠️ **Chưa đạt:**
- [TBD]

### 4.12.2. Key Findings

1. **Ticker performance:**
   - [TBD]

2. **Horizon performance:**
   - [TBD]

3. **Architecture comparison:**
   - [TBD]

4. **Optimization effectiveness:**
   - [TBD]

### 4.12.3. Recommendations

**For production:**
- Use models: [TBD]
- Avoid models: [TBD]
- Further testing needed: [TBD]

**For improvement:**
- [TBD]

---

**Chương tiếp theo:** Thảo luận và Kết luận

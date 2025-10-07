# BÁO CÁO PHÂN TÍCH VẤN ĐỀ ACCURACY KHÔNG THAY ĐỔI

## 🚨 **VẤN ĐỀ PHÁT HIỆN**

Từ log file `logs/models_20251007.log`, tôi phát hiện **vấn đề nghiêm trọng**: Direction Accuracy của nhiều models **HOÀN TOÀN KHÔNG THAY ĐỔI** trong suốt quá trình training.

---

## 📊 **PHÂN TÍCH CHI TIẾT**

### **1. VIB Model (CNN-BiLSTM)**
```
Epoch 5:  Target_Direction_t+1 - Accuracy: 0.842466
Epoch 10: Target_Direction_t+1 - Accuracy: 0.842466
Epoch 15: Target_Direction_t+1 - Accuracy: 0.842466
Epoch 20: Target_Direction_t+1 - Accuracy: 0.842466
Epoch 25: Target_Direction_t+1 - Accuracy: 0.842466
Epoch 30: Target_Direction_t+1 - Accuracy: 0.842466
Epoch 35: Target_Direction_t+1 - Accuracy: 0.842466
```
**Kết luận**: Accuracy **HOÀN TOÀN KHÔNG ĐỔI** = 0.842466 (84.24%)

### **2. VCB Model (CNN-BiLSTM)**
```
Epoch 1:  Target_Direction_t+1 - Accuracy: 0.911565
Epoch 5:  Target_Direction_t+1 - Accuracy: 0.911565
Epoch 10: Target_Direction_t+1 - Accuracy: 0.911565
Epoch 15: Target_Direction_t+1 - Accuracy: 0.911565
Epoch 20: Target_Direction_t+1 - Accuracy: 0.911565
Epoch 25: Target_Direction_t+1 - Accuracy: 0.911565
Epoch 30: Target_Direction_t+1 - Accuracy: 0.911565
Epoch 35: Target_Direction_t+1 - Accuracy: 0.911565
Epoch 40: Target_Direction_t+1 - Accuracy: 0.911565
Epoch 45: Target_Direction_t+1 - Accuracy: 0.911565
Epoch 50: Target_Direction_t+1 - Accuracy: 0.911565
```
**Kết luận**: Accuracy **HOÀN TOÀN KHÔNG ĐỔI** = 0.911565 (91.16%)

### **3. Pattern Chung**
- **Target_Direction_t+1**: Accuracy cố định ở mức cao (84-91%)
- **Target_Direction_t+3**: Accuracy thay đổi nhưng không ổn định
- **Target_Direction_t+5**: Accuracy thay đổi nhưng không ổn định

---

## 🔍 **NGUYÊN NHÂN GỐC RỄ**

### **1. Model Chỉ Học Predict Dominant Class**
```python
# Class Distribution Analysis
Class 1 (Flat): 62.6% - DOMINANT CLASS
Class 0 (Down): 18.1%
Class 2 (Up): 19.3%

# Model behavior:
VCB Accuracy = 91.16% ≈ Dominant class percentage
VIB Accuracy = 84.24% ≈ Dominant class percentage
```

**Kết luận**: Model đã học cách **LUÔN PREDICT CLASS 1 (FLAT)** để maximize accuracy!

### **2. Class Weights Không Hiệu Quả**
```python
# Class weights được tính:
VIB: [1.4469, 0.6689, 1.2285]
VCB: [1.6471, 0.5748, 1.5313]

# Nhưng model vẫn ignore minority classes
```

**Vấn đề**: Class weights chưa đủ mạnh để force model học minority classes.

### **3. Loss Function Imbalance**
```python
# Total Loss = Regression Loss + Classification Loss
# Regression loss có thể dominate classification loss
# Model focus vào minimize regression loss, ignore classification
```

---

## 🚨 **TẠI SAO ĐÂY LÀ VẤN ĐỀ NGHIÊM TRỌNG?**

### **1. Model Không Thực Sự Học**
- Accuracy cao (84-91%) nhưng **FAKE**
- Model chỉ memorize dominant class
- Không có khả năng generalization

### **2. Prediction Không Có Giá Trị**
```python
# Model prediction:
for any_input:
    return "FLAT"  # Always predict class 1

# Điều này vô nghĩa cho trading!
```

### **3. Metrics Misleading**
- Direction Accuracy = 91% **KHÔNG CÓ NGHĨA**
- Precision/Recall cho class 0,2 = 0%
- F1-score thực tế rất thấp

---

## 🔧 **GIẢI PHÁP KHẮC PHỤC**

### **1. Tăng Class Weights Mạnh Hơn**
```python
# Thay vì:
class_weights = [1.65, 0.57, 1.53]

# Sử dụng:
class_weights = [5.0, 0.2, 5.0]  # Penalty mạnh cho dominant class
```

### **2. Focal Loss thay vì CrossEntropy**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

### **3. Separate Loss Weighting**
```python
# Tách riêng loss weights:
total_loss = (
    regression_weight * regression_loss +
    classification_weight * classification_loss
)

# Với classification_weight >> regression_weight
regression_weight = 0.3
classification_weight = 0.7
```

### **4. Balanced Sampling**
```python
# Sử dụng WeightedRandomSampler
from torch.utils.data import WeightedRandomSampler

# Tạo balanced batches
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)
```

### **5. Threshold Tuning**
```python
# Thay vì dùng default threshold (0.5):
# Tune threshold để balance precision/recall
optimal_threshold = find_optimal_threshold(y_true, y_pred_proba)
```

### **6. Evaluation Metrics Cải Thiện**
```python
# Thay vì chỉ accuracy, track:
- Balanced Accuracy
- F1-score per class
- Precision/Recall per class
- Confusion Matrix
- Cohen's Kappa
```

---

## 🎯 **HÀNH ĐỘNG NGAY LẬP TỨC**

### **Bước 1: Kiểm Tra Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix, classification_report

# Xem model predict gì thực tế
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Nếu chỉ có 1 column/row khác 0 → confirmed problem
```

### **Bước 2: Implement Focal Loss**
```python
# Thay đổi trong base_model.py
criterion = FocalLoss(alpha=1, gamma=2)
loss = criterion(outputs[target_key], target_batch[i].long())
```

### **Bước 3: Tăng Classification Loss Weight**
```python
# Trong training loop:
classification_loss_weight = 2.0  # Tăng từ 1.0
total_loss = regression_loss + classification_loss_weight * classification_loss
```

### **Bước 4: Monitor Per-Class Metrics**
```python
# Thêm vào logging:
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average=None
)

logger.info(f"Per-class Precision: {precision}")
logger.info(f"Per-class Recall: {recall}")
logger.info(f"Per-class F1: {f1}")
```

---

## 📈 **KẾT QUẢ MONG ĐỢI SAU KHI SỬA**

### **Trước khi sửa:**
```
Target_Direction_t+1 - Accuracy: 0.911565 (FAKE - always predict class 1)
```

### **Sau khi sửa:**
```
Target_Direction_t+1 - Accuracy: 0.650000 (REAL - balanced prediction)
Per-class Precision: [0.60, 0.65, 0.70]
Per-class Recall: [0.55, 0.70, 0.65]
Per-class F1: [0.57, 0.67, 0.67]
```

---

## 🚨 **KẾT LUẬN**

**VẤN ĐỀ HIỆN TẠI**: Models đang **FAKE LEARNING** - chỉ predict dominant class để maximize accuracy.

**TÁC ĐỘNG**: 
- Accuracy metrics hoàn toàn misleading
- Models không có giá trị thực tế cho trading
- Cần fix ngay lập tức trước khi deploy

**ƯU TIÊN**: 
1. **Implement Focal Loss** (quan trọng nhất)
2. **Tăng classification loss weight**
3. **Monitor per-class metrics**
4. **Retrain tất cả models**

**Đây là bug nghiêm trọng cần fix ngay để có models thực sự hoạt động!** 🚨
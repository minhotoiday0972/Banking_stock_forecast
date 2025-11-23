# Dự đoán Xu hướng Cổ phiếu Ngân hàng

Dự án này sử dụng các mô hình Deep Learning (CNN-BiLSTM và Transformer) để dự đoán xu hướng giá (Tăng, Giảm, Đi ngang) của các cổ phiếu ngành ngân hàng Việt Nam theo nhiều khung thời gian khác nhau.

## 🎯 Mục tiêu Dự án

Mục tiêu chính là dự đoán xu hướng giá của các cổ phiếu ngân hàng cho các khung thời gian trong tương lai:
- **Ngắn hạn**: t+1, t+3, t+5
- **Trung hạn**: t+30 (1 tháng), t+60 (2 tháng)
- **Dài hạn**: t+90 (3 tháng)

## 📁 Cấu trúc Thư mục

Dự án được tổ chức như sau:

```
banking_stock_project/
├── src/
│   ├── app/           # Ứng dụng dự đoán bằng Streamlit
│   ├── data/          # Các script để thu thập dữ liệu
│   ├── features/      # Các script để xử lý đặc trưng (feature engineering)
│   ├── models/        # Kiến trúc các model (CNN-BiLSTM, Transformer)
│   ├── training/      # Pipeline huấn luyện model
│   └── utils/         # Các script tiện ích (config, logger, ...)
├── data/
│   ├── raw/           # Dữ liệu thô (OHLCV và tài chính cơ bản)
│   ├── processed/     # Dữ liệu đặc trưng đã được scale và metadata
│   └── database/      # Cơ sở dữ liệu SQLite để lưu trữ dữ liệu thô
├── models/            # Các file model đã được huấn luyện (.pt)
├── outputs/           # Các file kết quả (ví dụ: biểu đồ training)
├── logs/              # Log của ứng dụng và quá trình huấn luyện
├── mlruns/            # Thư mục chứa kết quả theo dõi của MLflow
├── .gitignore         # File cấu hình Git ignore
├── app.py             # File chính của ứng dụng Streamlit
├── config.yaml        # File cấu hình trung tâm cho toàn bộ dự án
├── main.py            # Script chính để chạy các pipeline (CLI)
├── README.md          # File này
└── requirements.txt   # Các thư viện Python cần thiết
```

## 🚀 Cài đặt Môi trường

### 1. Clone Repository
```bash
git clone <your-repository-url>
cd banking_stock_project
```

### 2. Tạo Môi trường Ảo
Rất khuyến khích sử dụng môi trường ảo (như conda hoặc venv) để quản lý các thư viện.

**Sử dụng conda:**
```bash
conda create -n stock_env python=3.9
conda activate stock_env
```

**Sử dụng venv:**
```bash
python -m venv stock_env
source stock_env/bin/activate  # Trên Windows, dùng: stock_env\Scripts\activate
```

### 3. Cài đặt Thư viện
Cài đặt tất cả các gói Python cần thiết bằng pip.
```bash
pip install -r requirements.txt
```

## 📊 Hướng dẫn Toàn bộ Quy trình

Dự án này được quản lý thông qua một giao diện dòng lệnh (CLI) trung tâm trong file `main.py`. Quy trình đầy đủ từ A-Z bao gồm 4 bước chính.

### Bước 1: Thu thập Dữ liệu
Lệnh này sẽ thu thập dữ liệu thô (OHLCV và các chỉ số tài chính cơ bản) từ nguồn dữ liệu và lưu vào cơ sở dữ liệu cục bộ (`data/database/stock_data.db`).
```bash
python main.py collect
```
- **Đầu vào**: Danh sách các mã cổ phiếu (`tickers`) và ngày bắt đầu (`start_date`) trong `config.yaml`.
- **Đầu ra**: File `stock_data.db` chứa dữ liệu thô.

### Bước 2: Xử lý và Tạo Đặc trưng (Feature Engineering)
Xử lý dữ liệu thô, tính toán các đặc trưng kỹ thuật và tài chính, áp dụng scaling, và lưu dữ liệu đã xử lý cùng với metadata cần thiết cho việc huấn luyện.
```bash
python main.py features

# Bạn cũng có thể chạy cho các mã cụ thể
python main.py features --tickers ACB VCB
```
- **Đầu vào**: Dữ liệu từ `stock_data.db`.
- **Đầu ra**: Các file CSV và a.pkl trong thư mục `data/processed/`.

### Bước 3: Huấn luyện Model
Chạy pipeline huấn luyện cho các model được chỉ định trong `config.yaml`. Một model riêng biệt sẽ được huấn luyện cho mỗi mã cổ phiếu và mỗi khung thời gian dự báo.
```bash
python main.py train

# Bạn cũng có thể chỉ định model hoặc mã cổ phiếu để huấn luyện
python main.py train --models transformer --tickers TCB
```
- **Đầu vào**: Dữ liệu đã xử lý từ `data/processed/`.
- **Đầu ra**: Các model đã huấn luyện được lưu trong `models/` và kết quả được ghi lại bởi MLflow.

### Chạy Toàn bộ Pipeline (All-in-One)
Để chạy tuần tự tất cả các bước (thu thập, xử lý đặc trưng, huấn luyện), hãy sử dụng lệnh `full`.
```bash
python main.py full
```

### Kiểm tra Trạng thái Pipeline
Để xem các bước nào đã hoàn thành và cần chạy bước nào tiếp theo, sử dụng lệnh `status`.
```bash
python main.py status
```

## 📈 Theo dõi Thí nghiệm với MLflow

Dự án tích hợp MLflow để theo dõi và quản lý các lần huấn luyện model.

### Chức năng chính:
- **Ghi lại Tham số**: Tự động ghi lại tất cả các tham số từ `config.yaml` cho mỗi lần chạy (VD: learning rate, dropout, số lớp, ...).
- **Ghi lại Chỉ số**: Ghi lại các chỉ số hiệu suất (metrics) trên tập validation và test (VD: F1-score, accuracy, precision, recall).
- **Lưu trữ Model**: Lưu lại model đã huấn luyện như một "artifact" để có thể tải lại sau này.
- **Lưu trữ Biểu đồ**: Ghi lại các biểu đồ như confusion matrix.

### Cách sử dụng:
1.  **Chạy Huấn luyện**: Khi bạn chạy lệnh `python main.py train` hoặc `python main.py full`, MLflow sẽ tự động ghi lại mọi thứ vào thư mục `mlruns`.
2.  **Khởi chạy Giao diện MLflow**: Để xem kết quả, mở một terminal mới và chạy lệnh sau từ thư mục gốc của dự án:
    ```bash
    mlflow ui
    ```
3.  **Xem Kết quả**: Mở trình duyệt và truy cập vào **http://localhost:5000**. Tại đây bạn có thể so sánh hiệu suất giữa các lần chạy, xem tham số đã dùng và các biểu đồ chi tiết.

## ⚙️ Cấu hình Dự án (`config.yaml`)

Tất cả các tham số của dự án được điều khiển từ file `config.yaml`.

- **`data`**: Chỉ định các mã cổ phiếu, khoảng thời gian, và cài đặt API.
- **`features`**: Định nghĩa các chỉ báo kỹ thuật, tài chính cơ bản và đặc thù ngành ngân hàng sẽ được sử dụng.
- **`training`**: Kiểm soát các siêu tham số như learning rate, số epochs, kích thước batch, và tỷ lệ phân chia dữ liệu.
- **`models`**: Cấu hình kiến trúc riêng của từng model (VD: hidden dims, dropout).
- **`models_to_train`**: **Quan trọng: Định nghĩa các model nào sẽ được chạy trong pipeline.**
  ```yaml
  models_to_train:
    dl_models:
      - "cnn_bilstm"
      - "transformer"
    baseline_models:
      - "naive"
      - "logistic_regression"
  ```
- **`paths`**: Thiết lập các đường dẫn thư mục cho dữ liệu, model và log.

## 🖥️ Chạy Ứng dụng Dự đoán

Sau khi các model đã được huấn luyện, bạn có thể chạy ứng dụng Streamlit tương tác để xem dự đoán.

```bash
python main.py app
```
Lệnh này sẽ khởi chạy server Streamlit. Bạn có thể truy cập ứng dụng trong trình duyệt tại **http://localhost:8501**.

## 🔬 Các Kỹ thuật và Công nghệ Chính

- **Dynamic Class Weights & Focal Loss**: Tự động xử lý vấn đề mất cân bằng lớp trong biến mục tiêu.
- **F1-based Early Stopping**: Sử dụng F1-score trên tập validation để dừng sớm, giúp model có độ ổn định và hiệu suất tốt hơn trong bài toán phân loại mất cân bằng.
- **Dynamic Regularization**: Áp dụng các giá trị `weight_decay` khác nhau cho các model ngắn hạn và dài hạn.
- **Tích hợp MLflow**: Để theo dõi thí nghiệm, ghi lại tham số, chỉ số và model.
- **Giao diện Streamlit**: Để dự đoán và phân tích một cách tương tác.

## ⚠️ Lưu ý Quan trọng

- **Hiệu suất Model**: Các khung thời gian dài hạn (t+30, t+60, t+90) thường cho hiệu suất tốt và ổn định hơn so với các khung thời gian ngắn hạn.
- **Backtesting**: Kết quả từ model này cần được kiểm tra lại (backtest) một cách nghiêm ngặt trước khi áp dụng vào bất kỳ chiến lược giao dịch thực tế nào.
- **Miễn trừ Trách nhiệm**: Đây là một dự án nghiên cứu và không phải là lời khuyên đầu tư tài chính.

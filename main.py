# main.py
"""
Script pipeline chính cho hệ thống dự báo chứng khoán
Bao gồm 2 nhánh:
1. (Mặc định) Dự đoán Ngắn hạn (t+1, t+3, t+5) dùng Deep Learning
2. (Quarterly) Dự đoán Dài hạn (t+1Q) dùng Deep Learning (LSTM)
"""
import argparse
import sys
import os

# Thêm thư mục src vào path để có thể import
sys.path.append(os.path.dirname(__file__))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.data.data_collector import DataCollector
from src.features.feature_engineer import FeatureEngineer
from src.training.trainer import ModelTrainingPipeline

# --- THÊM MỚI: Import 2 file của Nhánh 2 (Phiên bản DL) ---
from src.features.feature_engineer_quarterly import FeatureEngineerQuarterly
from src.training.train_quarterly import train_all_quarterly_models # Đổi tên hàm import

logger = get_logger("main")


def run_data_collection():
    """Chạy pipeline thu thập dữ liệu (Dùng chung cho cả 2 nhánh)"""
    print("\n" + "=" * 60)
    print("BƯỚC 1: THU THẬP DỮ LIỆU")
    print("=" * 60)
    logger.info("Bắt đầu thu thập dữ liệu...")
    collector = DataCollector()
    available, failed = collector.collect_all_data()
    print(f"\nKết quả Thu thập Dữ liệu:")
    print(f"  - Các mã có sẵn: {available}")
    if failed:
        print(f"  - Các mã thất bại: {failed}")
    if (len(available) + len(failed)) > 0:
        print(f"  - Tỷ lệ thành công: {len(available)}/{len(available) + len(failed)} ({len(available)/(len(available) + len(failed))*100:.1f}%)")
    return available, failed

# --- NHÁNH 1: DỰ ĐOÁN NGẮN HẠN (HÀNG NGÀY) ---

def run_feature_engineering(tickers=None):
    """(Nhánh 1) Chạy pipeline xử lý đặc trưng hàng ngày"""
    print("\n" + "=" * 60)
    print("BƯỚC 2 (NHÁNH 1): XỬ LÝ ĐẶC TRƯNG HÀNG NGÀY")
    print("=" * 60)
    logger.info("Bắt đầu xử lý đặc trưng hàng ngày...")
    engineer = FeatureEngineer()
    results = engineer.process_all_tickers(tickers)
    successful = [ticker for ticker, success in results.items() if success]
    failed = [ticker for ticker, success in results.items() if not success]
    print(f"\nKết quả Xử lý Đặc trưng (Hàng ngày):")
    print(f"  - Thành công: {successful}")
    if failed:
        print(f"  - Thất bại: {failed}")
    if results:
        print(f"  - Tỷ lệ thành công: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    return successful, failed


def run_model_training(model_types=None, tickers=None):
    """(Nhánh 1) Chạy pipeline huấn luyện model hàng ngày (Deep Learning)."""
    print("\n" + "=" * 60)
    print("BƯỚC 3 (NHÁNH 1): HUẤN LUYỆN MODEL HÀNG NGÀY (DEEP LEARNING)")
    print("=" * 60)
    logger.info("Bắt đầu huấn luyện model hàng ngày...")
    pipeline = ModelTrainingPipeline()
    results = pipeline.train_all_models(model_types, tickers)
    print(f"\n" + "=" * 60)
    print("TỔNG KẾT HUẤN LUYỆN (HÀNG NGÀY)")
    print("=" * 60)
    total_success = 0
    total_attempted = 0
    for model_type, model_results in results.items():
        successful_models = [t for t, r in model_results.items() if r and r.get('test_metrics')]
        total_tried = len(model_results)
        success_count = len(successful_models)
        total_success += success_count
        total_attempted += total_tried
        print(f"--- Model: {model_type.upper()} ---")
        print(f"  Hoàn thành: {success_count}/{total_tried} mã cổ phiếu.")
    print("\nToàn bộ quá trình huấn luyện hàng ngày đã hoàn tất.")
    print(f"Tổng cộng: {total_success}/{total_attempted} model đã được huấn luyện thành công.")
    print(f"Chi tiết kết quả đã được lưu vào file log (ví dụ: logs/trainer.log).")
    return results


def run_full_pipeline_daily(model_types=None):
    """(Nhánh 1) Chạy toàn bộ pipeline hàng ngày."""
    print("\n>>> PIPELINE DỰ BÁO NGẮN HẠN (HÀNG NGÀY) <<<")
    logger.info("Bắt đầu chạy toàn bộ pipeline hàng ngày...")
    available_tickers, _ = run_data_collection()
    if not available_tickers:
        logger.error("Không có dữ liệu nào được thu thập. Dừng pipeline.")
        return
    successful_features, _ = run_feature_engineering(available_tickers)
    if not successful_features:
        logger.error("Không có đặc trưng nào được xử lý. Dừng pipeline.")
        return
    run_model_training(model_types, successful_features)
    print("\n" + "=" * 60)
    print("PIPELINE HÀNG NGÀY HOÀN THÀNH!")
    print("=" * 60)


# --- CẬP NHẬT: NHÁNH 2 (DỰ ĐOÁN DÀI HẠN / HÀNG QUÝ - PHIÊN BẢN DL) ---

def run_feature_engineering_quarterly(tickers=None):
    """(Nhánh 2) Chạy pipeline xử lý đặc trưng hàng quý (Phiên bản DL)."""
    print("\n" + "=" * 60)
    print("BƯỚC 2 (NHÁNH 2): XỬ LÝ ĐẶC TRƯNG HÀNG QUÝ (CHO DL)")
    print("=" * 60)
    logger.info("Bắt đầu xử lý đặc trưng hàng quý...")
    engineer = FeatureEngineerQuarterly()
    engineer.process_all_tickers(tickers)
    print("\nKết quả Xử lý Đặc trưng (Hàng quý): Hoàn thành (Xem log để biết chi tiết).")

def run_model_training_quarterly():
    """(Nhánh 2) Chạy pipeline huấn luyện model hàng quý (LSTM)."""
    # Hàm train_all_quarterly_models đã bao gồm tiêu đề
    train_all_quarterly_models()

def run_full_pipeline_quarterly():
    """(Nhánh 2) Chạy toàn bộ pipeline hàng quý (Phiên bản DL)."""
    print("\n>>> PIPELINE DỰ BÁO DÀI HẠN (HÀNG QUÝ) <<<")
    logger.info("Bắt đầu chạy toàn bộ pipeline hàng quý...")
    
    available_tickers, _ = run_data_collection()
    if not available_tickers:
        logger.error("Không có dữ liệu nào được thu thập. Dừng pipeline.")
        return

    run_feature_engineering_quarterly(available_tickers)
    run_model_training_quarterly()
    
    print("\n" + "=" * 60)
    print("PIPELINE HÀNG QUÝ HOÀN THÀNH!")
    print("=" * 60)

# --- CÁC HÀM TIỆN ÍCH (Cập nhật run_status_check) ---

def run_status_check():
    """Kiểm tra trạng thái của cả hai nhánh pipeline"""
    print("\n" + "=" * 60)
    print("KIỂM TRA TRẠNG THÁI PIPELINE")
    print("=" * 60)

    config = get_config()
    expected_tickers = config.get('data.tickers', [])
    model_configs = config.get('models', {})
    expected_models_dl = [m for m in model_configs.keys() if m != 'shared']

    # Tạo thư mục
    os.makedirs(config.get('paths.database', 'data/database'), exist_ok=True)
    os.makedirs(config.get('paths.processed', 'data/processed'), exist_ok=True)
    os.makedirs(config.get('paths.models', 'models'), exist_ok=True)

    # 1. Thu thập dữ liệu (Chung)
    db_file = os.path.join(config.get('paths.database', 'data/database'), 'stock_data.db')
    data_status = os.path.exists(db_file)
    print(f"1. Thu thập dữ liệu (Chung):    {'HOÀN THÀNH' if data_status else 'CHƯA CÓ'}")

    # --- NHÁNH 1: DỰ ĐOÁN NGẮN HẠN (DEEP LEARNING) ---
    print("\n--- NHÁNH 1: DỰ ĐOÁN NGẮN HẠN (DEEP LEARNING) ---")
    processed_dir = config.get('paths.processed', 'data/processed')
    models_dir = config.get('paths.models', 'models')

    # Initialize statuses for Nhánh 1
    features_status_dl = False
    models_status_dl = False
    app_status = False # <<< FIX: Initialize app_status here

    feature_files_dl = [f for f in os.listdir(processed_dir) if f.endswith("_metadata.pkl")] if os.path.exists(processed_dir) else []
    features_status_dl = len(feature_files_dl) >= len(expected_tickers)
    print(f"2a. Xử lý đặc trưng (Hàng ngày): {'HOÀN THÀNH' if features_status_dl else 'CHƯA CÓ'} ({len(feature_files_dl)}/{len(expected_tickers)} file metadata)")

    if features_status_dl: # Only check models if features exist
        model_files_dl = [f for f in os.listdir(models_dir) if f.endswith(".pt") and not "quarterly" in f] if os.path.exists(models_dir) else []
        expected_model_count_dl = len(expected_tickers) * len(expected_models_dl)
        models_status_dl = len(model_files_dl) >= expected_model_count_dl
        print(f"3a. Huấn luyện model (Hàng ngày):   {'HOÀN THÀNH' if models_status_dl else 'CHƯA CÓ'} ({len(model_files_dl)}/{expected_model_count_dl} model)")

        if models_status_dl: # Only check app if models exist
             app_status = os.path.exists("app.py")
             print(f"4a. Sẵn sàng của App (Hàng ngày):   {'SẴN SÀNG' if app_status else 'CHƯA SẴN SÀNG'}")
        else:
             print(f"4a. Sẵn sàng của App (Hàng ngày):   CHƯA SẴN SÀNG (Thiếu model)")
    else:
        print(f"3a. Huấn luyện model (Hàng ngày):   CHƯA CÓ (Thiếu đặc trưng)")
        print(f"4a. Sẵn sàng của App (Hàng ngày):   CHƯA SẴN SÀNG (Thiếu đặc trưng)")


    # --- NHÁNH 2: DỰ ĐOÁN DÀI HẠN (DEEP LEARNING) ---
    print("\n--- NHÁNH 2: DỰ ĐOÁN DÀI HẠN (DEEP LEARNING) ---")
    # Initialize statuses for Nhánh 2
    features_status_q = False
    models_status_q = False

    feature_files_q = [f for f in os.listdir(processed_dir) if f.endswith("_quarterly_features.csv")] if os.path.exists(processed_dir) else []
    features_status_q = len(feature_files_q) >= len(expected_tickers)
    print(f"2b. Xử lý đặc trưng (Hàng quý):  {'HOÀN THÀNH' if features_status_q else 'CHƯA CÓ'} ({len(feature_files_q)}/{len(expected_tickers)} file)")

    if features_status_q: # Only check models if features exist
        model_files_q = [f for f in os.listdir(models_dir) if f.endswith("_quarterly_cnn_bilstm.pt") or f.endswith("_quarterly_transformer.pt")] if os.path.exists(models_dir) else []
        expected_model_count_q = len(expected_tickers) * len(expected_models_dl) # Uses same model types
        models_status_q = len(model_files_q) >= expected_model_count_q
        print(f"3b. Huấn luyện model (Hàng quý):    {'HOÀN THÀNH' if models_status_q else 'CHƯA CÓ'} ({len(model_files_q)}/{expected_model_count_q} model)")
    else:
         print(f"3b. Huấn luyện model (Hàng quý):    CHƯA CÓ (Thiếu đặc trưng)")

    # --- Đề xuất bước tiếp theo ---
    print("\n=> BƯỚC TIẾP THEO ĐƯỢC ĐỀ XUẤT:")
    if not data_status:
        print("     python main.py collect")
    elif not features_status_dl:
        print("     python main.py features")
    elif not features_status_q:
         print("     python main.py features_quarterly")
    elif not models_status_dl:
        print("     python main.py train --models all")
    elif not models_status_q:
        print("     python main.py train_quarterly")
    elif app_status: # Only suggest app if Nhánh 1 is fully ready
        print("     python main.py app")
    else:
        # Fallback if app isn't ready but everything else is
        print("     Kiểm tra file app.py hoặc chạy lại huấn luyện Nhánh 1.")


def run_app():
    """Khởi chạy ứng dụng Streamlit (Chỉ cho Nhánh 1)"""
    print("\n" + "=" * 60)
    print("KHỞI CHẠY ỨNG DỤNG DỰ BÁO (NGẮN HẠN)")
    print("=" * 60)
    print("Ứng dụng sẽ có tại: http://localhost:8501")
    print("Nhấn Ctrl+C để dừng ứng dụng")
    import subprocess
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except FileNotFoundError:
        logger.error("Lệnh `streamlit` không tồn tại. Hãy đảm bảo Streamlit đã được cài đặt (`pip install streamlit`).")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi chạy ứng dụng Streamlit: {e}")
    except KeyboardInterrupt:
        print("\nỨng dụng đã được dừng bởi người dùng.")

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description="Banking Stock Prediction Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Các ví dụ sử dụng:
  python main.py collect                  # Chỉ thu thập dữ liệu (Dùng chung)
  python main.py status                   # Kiểm tra trạng thái của cả 2 nhánh
  
  --- NHÁNH 1: DỰ ĐOÁN NGẮN HẠN (HÀNG NGÀY) ---
  python main.py features                 # Xử lý đặc trưng hàng ngày
  python main.py train --models all       # Huấn luyện model Deep Learning hàng ngày
  python main.py full_daily               # Chạy toàn bộ nhánh 1
  python main.py app                      # Khởi chạy ứng dụng (Nhánh 1)

  --- NHÁNH 2: DỰ ĐOÁN DÀI HẠN (HÀNG QUÝ) ---
  python main.py features_quarterly       # Xử lý đặc trưng hàng quý (cho DL)
  python main.py train_quarterly          # Huấn luyện model (CNN-LSTM, Trans) hàng quý
  python main.py full_quarterly           # Chạy toàn bộ nhánh 2
"""
    )
    
    commands = [
        "collect", "status", "app",
        "features", "train", "full_daily",
        "features_quarterly", "train_quarterly", "full_quarterly"
    ]
    parser.add_argument("command", choices=commands, help="Bước pipeline cần chạy")
    
    parser.add_argument("--models", nargs="+", default=None, help="[Nhánh 1] Các model DL cần huấn luyện (ví dụ: cnn_bilstm transformer). Mặc định là tất cả.")
    parser.add_argument("--tickers", nargs="+", default=None, help="[Nhánh 1 & 2] Các mã ticker cụ thể để xử lý.")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn file config.")

    args = parser.parse_args()
    
    if args.models and "all" in args.models:
        config = get_config(args.config)
        all_model_configs = config.get('models', {})
        args.models = [model_name for model_name in all_model_configs.keys() if model_name != 'shared']

    try:
        if args.command == "collect":
            run_data_collection()
        elif args.command == "status":
            run_status_check()
        elif args.command == "app":
            run_app()
            
        elif args.command == "features":
            run_feature_engineering(args.tickers)
        elif args.command == "train":
            run_model_training(args.models, args.tickers)
        elif args.command == "full_daily":
            run_full_pipeline_daily(args.models)

        elif args.command == "features_quarterly":
            run_feature_engineering_quarterly(args.tickers)
        elif args.command == "train_quarterly":
            run_model_training_quarterly()
        elif args.command == "full_quarterly":
            run_full_pipeline_quarterly()

    except Exception as e:
        logger.error(f"Một lỗi đã xảy ra trong pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
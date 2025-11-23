# main.py
"""
Script pipeline chính cho hệ thống dự báo chứng khoán
ĐÃ TÁI CẤU TRÚC: Chỉ tập trung vào Nhánh 1 (Hàng ngày)
với các mô hình chuyên biệt cho từng tầm nhìn.
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
# --- XÓA: Import cho Nhánh 2 ---

logger = get_logger("main", log_filename="main_training.log")

def run_data_collection():
    """Chạy pipeline thu thập dữ liệu"""
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
    return available, failed


def run_feature_engineering(tickers=None):
    """BƯỚC 2: XỬ LÝ ĐẶC TRƯNG"""
    print("\n" + "=" * 60)
    print("BƯỚC 2: XỬ LÝ ĐẶC TRƯNG (Tạo Metadata Chuyên biệt)")
    print("=" * 60)
    logger.info("Bắt đầu xử lý đặc trưng (tự động lựa chọn)...")

    engineer = FeatureEngineer()
    results = engineer.process_all_tickers(tickers)

    successful = [ticker for ticker, success in results.items() if success]
    failed = [ticker for ticker, success in results.items() if not success]

    print(f"\nKết quả Xử lý Đặc trưng:")
    print(f"  - Thành công: {successful}")
    if failed:
        print(f"  - Thất bại: {failed}")
    if results:
        print(f"  - Tỷ lệ thành công: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    return successful, failed


def run_model_training(model_types=None, tickers=None):
    """BƯỚC 3: HUẤN LUYỆN MODEL CHUYÊN BIỆT"""
    print("\n" + "=" * 60)
    print("BƯỚC 3: HUẤN LUYỆN MODEL CHUYÊN BIỆT")
    print("=" * 60)
    logger.info("Bắt đầu huấn luyện các model chuyên biệt...")

    pipeline = ModelTrainingPipeline()
    results = pipeline.train_all_models(model_types, tickers)
    
    print(f"\n" + "=" * 60)
    print("TỔNG KẾT HUẤN LUYỆN")
    print("=" * 60)
    
    if not results:
        print("Không có kết quả nào được trả về từ quá trình huấn luyện.")
        return results

    successful_models = [k for k, v in results.items() if v and v.get('balanced_accuracy', 0) > 0.5]
    failed_models = [k for k, v in results.items() if not v or v.get('balanced_accuracy', 0) <= 0.5]
    
    print(f"  - {len(successful_models)} model huấn luyện thành công (Bal_Acc > 50%)")
    print(f"  - {len(failed_models)} model thất bại hoặc hiệu suất thấp.")
    print("Chi tiết kết quả đã được lưu vào file log (logs/trainer.log).")
    return results


def run_full_pipeline(model_types=None, tickers=None):
    """Chạy toàn bộ pipeline (Collect, Features, Train)"""
    print("\n>>> PIPELINE DỰ BÁO XU HƯỚNG CỔ PHIẾU <<<")
    logger.info("Bắt đầu chạy toàn bộ pipeline...")

    available_tickers, _ = run_data_collection()
    if not available_tickers:
        logger.error("Dừng: Không có dữ liệu.")
        return
        
    # Lọc tickers nếu người dùng chỉ định
    if tickers:
        available_tickers = [t for t in tickers if t in available_tickers]
        if not available_tickers:
             logger.error(f"Dừng: Các tickers chỉ định {tickers} không có trong dữ liệu đã thu thập.")
             return

    successful_features, _ = run_feature_engineering(available_tickers)
    if not successful_features:
        logger.error("Dừng: Không xử lý được đặc trưng.")
        return

    run_model_training(model_types, successful_features)
    print("\n" + "=" * 60)
    print("PIPELINE HOÀN THÀNH!")
    print("=" * 60)
    print(f"Sẵn sàng để chạy ứng dụng: streamlit run app.py")


def run_status_check():
    """Kiểm tra trạng thái của pipeline"""
    print("\n" + "=" * 60)
    print("KIỂM TRA TRẠNG THÁI PIPELINE")
    print("=" * 60)
    
    config = get_config()
    expected_tickers = config.get('data.tickers', [])
    model_configs = config.get('models', {})
    expected_models = [m for m in model_configs.keys() if m != 'shared']
    expected_horizons = config.get('models.shared.forecast_horizons', [])
    
    # Tạo thư mục
    os.makedirs(config.get('paths.database', 'data/database'), exist_ok=True)
    os.makedirs(config.get('paths.processed', 'data/processed'), exist_ok=True)
    os.makedirs(config.get('paths.models', 'models'), exist_ok=True)

    # 1. Thu thập dữ liệu
    db_file = os.path.join(config.get('paths.database', 'data/database'), 'stock_data.db')
    data_status = os.path.exists(db_file)
    print(f"1. Thu thập dữ liệu:    {'HOÀN THÀNH' if data_status else 'CHƯA CÓ'}")

    # 2. Xử lý đặc trưng
    processed_dir = config.get('paths.processed', 'data/processed')
    # Kiểm tra file metadata cho mỗi horizon
    expected_metadata_count = len(expected_tickers) * len(expected_horizons)
    feature_files = [f for f in os.listdir(processed_dir) if f.endswith(".pkl") and "metadata_t+" in f] if os.path.exists(processed_dir) else []
    features_status = len(feature_files) >= expected_metadata_count
    print(f"2. Xử lý đặc trưng:      {'HOÀN THÀNH' if features_status else 'CHƯA CÓ'} ({len(feature_files)}/{expected_metadata_count} file metadata)")

    # 3. Huấn luyện model
    models_dir = config.get('paths.models', 'models')
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt") and "_t+" in f] if os.path.exists(models_dir) else []
    expected_model_count = len(expected_tickers) * len(expected_models) * len(expected_horizons)
    models_status = len(model_files) >= expected_model_count
    print(f"3. Huấn luyện model:      {'HOÀN THÀNH' if models_status else 'CHƯA CÓ'} ({len(model_files)}/{expected_model_count} model)")

    # 4. Sẵn sàng của App
    app_status = os.path.exists("app.py") and models_status
    print(f"4. Sẵn sàng của App:      {'SẴN SÀNG' if app_status else 'CHƯA SẴN SÀNG'}")

    # --- XÓA: Kiểm tra Nhánh 2 ---

    print(f"\n=> BƯỚC TIẾP THEO ĐƯỢC ĐỀ XUẤT:")
    if not data_status: print("     python main.py collect")
    elif not features_status: print("     python main.py features")
    elif not models_status: print(f"     python main.py train --models all")
    elif app_status: print("     streamlit run app.py")


def run_app():
    """Khởi chạy ứng dụng Streamlit"""
    print("\n" + "=" * 60)
    print("KHỞI CHẠY ỨNG DỤNG DỰ BÁO")
    print("=" * 60)
    print("Ứng dụng sẽ có tại: http://localhost:8501")
    print("Nhấn Ctrl+C để dừng ứng dụng")
    import subprocess
    try:
        # --- THAY ĐỔI: Chạy streamlit từ subprocess ---
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except FileNotFoundError:
        logger.error("Lệnh `streamlit` không tồn tại. Hãy đảm bảo Streamlit đã được cài đặt (`pip install streamlit`).")
    except Exception as e:
        logger.error(f"Lỗi khi chạy 'streamlit run app.py': {e}")
    # --- KẾT THÚC THAY ĐỔI ---


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description="Banking Stock Prediction Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Các ví dụ sử dụng:
  python main.py collect                  # Chỉ thu thập dữ liệu
  python main.py features                 # Xử lý đặc trưng (cho tất cả tickers)
  python main.py features --tickers ACB   # Xử lý đặc trưng (chỉ cho ACB)
  python main.py train --models all       # Huấn luyện tất cả model (cho tất cả tickers)
  python main.py train --models transformer --tickers TCB VPB  # Huấn luyện model cụ thể cho ticker cụ thể
  python main.py full                     # Chạy toàn bộ pipeline
  python main.py status                   # Kiểm tra trạng thái
  python main.py app                      # Khởi chạy ứng dụng
"""
    )
    
    # --- THAY ĐỔI: Loại bỏ các lệnh của Nhánh 2 ---
    commands = ["collect", "features", "train", "full", "status", "app"]
    parser.add_argument("command", choices=commands, help="Bước pipeline cần chạy")
    
    parser.add_argument("--models", nargs="+", default=None, help="Các model DL cần huấn luyện (ví dụ: cnn_bilstm transformer).")
    parser.add_argument("--tickers", nargs="+", default=None, help="Các mã ticker cụ thể để xử lý.")
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn file config.")

    args = parser.parse_args()
    
    if args.models and "all" in args.models:
        config = get_config(args.config)
        all_model_configs = config.get('models', {})
        args.models = [model_name for model_name in all_model_configs.keys() if model_name != 'shared']

    try:
        if args.command == "collect":
            run_data_collection()
        elif args.command == "features":
            run_feature_engineering(args.tickers)
        elif args.command == "train":
            run_model_training(args.models, args.tickers)
        elif args.command == "full":
            # --- THAY ĐỔI: Truyền tickers vào full pipeline ---
            run_full_pipeline(args.models, args.tickers)
        elif args.command == "status":
            run_status_check()
        elif args.command == "app":
            run_app()

    except Exception as e:
        logger.error(f"Một lỗi đã xảy ra trong pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
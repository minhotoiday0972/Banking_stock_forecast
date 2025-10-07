# main.py
"""
Main pipeline script for stock prediction system
"""
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.data.data_collector import DataCollector
from src.features.feature_engineer import FeatureEngineer
from src.training.trainer import ModelTrainingPipeline

logger = get_logger("main")


def run_data_collection():
    """Run data collection pipeline"""
    print("\n" + "=" * 60)
    print("DATA COLLECTION")
    print("=" * 60)
    logger.info("Starting data collection...")

    collector = DataCollector()
    available, failed = collector.collect_all_data()

    print(f"\nData Collection Results:")
    print(f"Available tickers: {available}")
    if failed:
        print(f"Failed tickers: {failed}")
    print(
        f"Success rate: {len(available)}/{len(available) + len(failed)} ({len(available)/(len(available) + len(failed))*100:.1f}%)"
    )

    return available, failed


def run_feature_engineering(tickers=None):
    """Run feature engineering pipeline"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    logger.info("Starting feature engineering...")

    engineer = FeatureEngineer()
    results = engineer.process_all_tickers(tickers)

    successful = [ticker for ticker, success in results.items() if success]
    failed = [ticker for ticker, success in results.items() if not success]

    print(f"\nFeature Engineering Results:")
    print(f"Successful: {successful}")
    if failed:
        print(f"Failed: {failed}")
    print(
        f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)"
    )

    return successful, failed


def run_model_training(model_types=None, tickers=None):
    """Run model training pipeline"""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    logger.info("Starting model training...")

    pipeline = ModelTrainingPipeline()
    results = pipeline.train_all_models(model_types, tickers)

    print(f"\nModel Training Results:")
    for model_type, model_results in results.items():
        successful_models = [t for t, r in model_results.items() if r is not None]
        print(f"{model_type}: {len(successful_models)} models trained")

    return results


def run_full_pipeline(model_types=None):
    """Run complete pipeline from data collection to model training"""
    print("\nBANKING STOCK PREDICTION PIPELINE")
    print("=" * 60)
    logger.info("Starting full pipeline...")

    # Step 1: Data Collection
    available_tickers, failed_collection = run_data_collection()
    if not available_tickers:
        logger.error("No data collected successfully. Stopping pipeline.")
        return

    # Step 2: Feature Engineering
    successful_features, failed_features = run_feature_engineering(available_tickers)
    if not successful_features:
        logger.error("No features engineered successfully. Stopping pipeline.")
        return

    # Step 3: Model Training
    training_results = run_model_training(model_types, successful_features)

    # Final Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(
        f"Data Collection: {len(available_tickers)} successful, {len(failed_collection)} failed"
    )
    print(
        f"Feature Engineering: {len(successful_features)} successful, {len(failed_features)} failed"
    )

    total_models = 0
    for model_type, model_results in training_results.items():
        successful_models = [t for t, r in model_results.items() if r is not None]
        total_models += len(successful_models)
        print(f"{model_type}: {len(successful_models)} models")

    print(f"\nPIPELINE COMPLETED!")
    print(
        f"Total: {total_models} models trained for {len(successful_features)} banks"
    )
    print(f"Ready to run: streamlit run app.py")

    logger.info("Full pipeline completed successfully")


def run_status_check():
    """Check pipeline status"""
    print("\n" + "=" * 60)
    print("PIPELINE STATUS CHECK")
    print("=" * 60)

    import os

    # Create directories if needed
    os.makedirs("data/database", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Check data collection
    db_file = "data/database/stock_data.db"
    data_status = os.path.exists(db_file)
    print(f"Data Collection: {'Complete' if data_status else 'Missing'}")

    # Check feature engineering
    processed_dir = "data/processed"
    feature_files = (
        [f for f in os.listdir(processed_dir) if f.endswith("_features.csv")]
        if os.path.exists(processed_dir)
        else []
    )
    features_status = len(feature_files) >= 10
    print(
        f"Feature Engineering: {'Complete' if features_status else 'Missing'} ({len(feature_files)} files)"
    )

    # Check model training
    models_dir = "models"
    model_files = (
        [f for f in os.listdir(models_dir) if f.endswith(".pt")]
        if os.path.exists(models_dir)
        else []
    )
    models_status = len(model_files) >= 30
    print(
        f"Model Training: {'Complete' if models_status else 'Missing'} ({len(model_files)} models)"
    )

    # Check app readiness
    app_status = os.path.exists("app.py") and models_status
    print(f"App Ready: {'Ready' if app_status else 'Not Ready'}")

    print(f"\nNEXT STEPS:")
    if not data_status:
        print("   python main.py collect")
    elif not features_status:
        print("   python main.py features")
    elif not models_status:
        print("   python main.py train --models all")
    elif app_status:
        print("   streamlit run app.py")
    else:
        print("   Check app.py file exists")


def run_app():
    """Launch the Streamlit app"""
    print("\n" + "=" * 60)
    print("LAUNCHING PREDICTION APP")
    print("=" * 60)
    print("App will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the app")

    import subprocess

    try:
        subprocess.run("streamlit run app.py", shell=True)
    except KeyboardInterrupt:
        print("\nApp stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Banking Stock Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py collect              # Collect banking data
  python main.py features             # Engineer features
  python main.py train --models all   # Train all models
  python main.py full                 # Run complete pipeline
  python main.py app                  # Launch prediction app
  python main.py status               # Check pipeline status
        """,
    )
    parser.add_argument(
        "command",
        choices=["collect", "features", "train", "full", "status", "app"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn_bilstm"],
        choices=["cnn_bilstm", "transformer", "all"],
        help="Models to train (for train/full commands)",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None, help="Specific tickers to process"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")

    args = parser.parse_args()

    # Handle 'all' models option
    if args.models and "all" in args.models:
        args.models = ["cnn_bilstm", "transformer"]

    try:
        # Load config
        config = get_config(args.config)
        logger.info(f"Loaded config from {args.config}")

        # Run requested command
        if args.command == "collect":
            run_data_collection()
        elif args.command == "features":
            run_feature_engineering(args.tickers)
        elif args.command == "train":
            run_model_training(args.models, args.tickers)
        elif args.command == "full":
            run_full_pipeline(args.models)
        elif args.command == "status":
            run_status_check()
        elif args.command == "app":
            run_app()

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

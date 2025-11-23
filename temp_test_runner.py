# temp_test_runner.py
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(__file__))

from src.utils.logger import get_logger
from src.data.data_collector import DataCollector
from src.features.feature_engineer import FeatureEngineer
from src.training.trainer import ModelTrainingPipeline

logger = get_logger("test_runner")

def run_test_pipeline():
    """
    Runs a minimal pipeline to test the changes.
    1. Collect data
    2. Engineer features
    3. Train transformer model
    """
    try:
        logger.info("========== STARTING TEST RUNNER ==========")

        # 1. Data Collection
        logger.info("--- Step 1: Data Collection ---")
        collector = DataCollector()
        available_tickers, _ = collector.collect_all_data()
        if not available_tickers:
            logger.error("Data collection failed. Aborting.")
            return

        # 2. Feature Engineering
        logger.info("--- Step 2: Feature Engineering ---")
        engineer = FeatureEngineer()
        successful_features, _ = engineer.process_all_tickers(available_tickers)
        if not successful_features:
            logger.error("Feature engineering failed. Aborting.")
            return

        # 3. Model Training (Transformer only)
        logger.info("--- Step 3: Model Training (Transformer) ---")
        training_pipeline = ModelTrainingPipeline()
        results = training_pipeline.train_all_models(model_types=['transformer'], tickers=successful_features)

        logger.info("========== TEST RUNNER FINISHED ==========")
        
        if results:
            logger.info("Test run completed. Check trainer logs for detailed results.")
        else:
            logger.warning("Test run finished, but no results were generated.")

    except Exception as e:
        logger.error(f"An error occurred in the test runner: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_test_pipeline()

#!/usr/bin/env python3
"""
Full Banking Stock Prediction Pipeline
Runs complete pipeline from data collection to model training
"""
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
import yaml

def log_step(step_name, status="START"):
    """Log pipeline step with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"[{timestamp}] {status}: {step_name}")
    print(f"{'='*60}")

def run_command(command, description):
    """Run command and handle errors"""
    log_step(description)
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
            return True
        else:
            print(f"FAILED: {description}")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False



def check_prerequisites():
    """Check if all prerequisites are met"""
    log_step("Checking Prerequisites")
    
    # Check Python environment
    print(f"Python version: {sys.version}")
    
    # Check required files
    required_files = [
        "config.yaml",
        "main.py",
        "src/data/data_collector.py",
        "src/features/feature_engineer.py",
        "src/training/trainer.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return False
    
    # Check directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("All prerequisites met")
    return True

def run_full_pipeline():
    """Run the complete pipeline"""
    start_time = time.time()
    
    print("STARTING FULL BANKING STOCK PREDICTION PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("Prerequisites check failed. Exiting.")
        return False
    
    # Step 2: Data collection (automatically uses latest date)
    print("Data collection will automatically use latest available date")
    
    # Step 3: Collect data
    if not run_command("python main.py collect", "Data Collection"):
        print("Data collection failed. Exiting.")
        return False
    
    # Step 4: Feature engineering
    if not run_command("python main.py features", "Feature Engineering"):
        print("Feature engineering failed. Exiting.")
        return False
    
    # Step 5: Train models (all models)
    models = ["cnn_bilstm", "transformer"]
    
    for model in models:
        if not run_command(f"python main.py train --models {model}", f"Training {model.upper()} Model"):
            print(f"{model.upper()} training failed, continuing with next model...")
            continue
    
    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    log_step("PIPELINE COMPLETED", "FINISH")
    print(f"Total time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return True

def run_quick_pipeline():
    """Run pipeline with only transformer model (faster)"""
    start_time = time.time()
    
    print("🚀 STARTING QUICK PIPELINE (Transformer only)")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Prerequisites check
    if not check_prerequisites():
        return False
    
    print("📅 Data collection will automatically use latest available date")
    
    # Quick pipeline steps
    steps = [
        ("python main.py collect", "Data Collection"),
        ("python main.py features", "Feature Engineering"),
        ("python main.py train --models transformer", "Training Transformer Model"),
        ("python main.py predict", "Generating Predictions")
    ]
    
    for command, description in steps:
        if not run_command(command, description):
            print(f"❌ {description} failed. Exiting.")
            return False
    
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    log_step("QUICK PIPELINE COMPLETED", "FINISH")
    print(f"⏱️ Total time: {minutes:02d}:{seconds:02d}")
    
    return True

def main():
    """Main function with options"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            success = run_quick_pipeline()
        elif sys.argv[1] == "full":
            success = run_full_pipeline()
        else:
            print("Usage: python run_full_pipeline.py [full|quick]")
            print("  full  - Run complete pipeline with all models")
            print("  quick - Run pipeline with transformer only")
            return
    else:
        # Default to full pipeline
        success = run_full_pipeline()
    
    if success:
        print("\nPipeline completed successfully!")
        print("Run 'python check_results.py' to view results")
    else:
        print("\nPipeline failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
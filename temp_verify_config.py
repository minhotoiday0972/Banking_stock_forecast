import sys
import os
import torch

# Add the project root to the sys.path
sys.path.insert(0, os.path.abspath('.'))

from src.training.trainer import ModelTrainingPipeline
from src.utils.config import get_config

def verify_pipeline_config():
    print("--- Verifying ModelTrainingPipeline Configuration ---")

    # Initialize the pipeline without actually running training
    pipeline = ModelTrainingPipeline()

    print("\nModels to train (from pipeline initialization):")
    print(list(pipeline.models_to_train.keys()))

    print("\nTraining configuration related to class weights:")
    training_config = get_config().get('training', {})
    print(f"use_focal_loss: {training_config.get('use_focal_loss')}")
    print(f"use_manual_class_weights: {training_config.get('use_manual_class_weights')}")
    print(f"manual_class_weights (should be None/removed): {training_config.get('manual_class_weights')}")

    # Optional: Test class weights calculation (requires data, so skipping for now)
    # print("\nTesting dynamic class weights calculation (requires data, skipping)...")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_pipeline_config()

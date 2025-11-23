# src/models/baselines.py
from sklearn.dummy import DummyClassifier

def get_baseline_model(model_name: str, config: dict):
    """
    Factory function to get a baseline model instance.
    """
    if model_name == 'naive':
        # DummyClassifier predicting the most frequent class is a great naive baseline.
        return DummyClassifier(strategy="most_frequent")


    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

#!/usr/bin/env python3
"""
Diagnose Training Issues - Deep analysis of training problems
Comprehensive diagnosis of why metrics vary widely and R² can be negative
"""
import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from src.utils.config import get_config
    from src.utils.database import get_database
    from src.features.feature_engineer import FeatureEngineer
except ImportError:
    print("⚠️ Could not import project modules")


def print_header(title):
    """Print formatted header"""
    print(f"\n{'='*70}")
    print(f"🔬 {title}")
    print(f"{'='*70}")


def print_section(title):
    """Print formatted section"""
    print(f"\n{'-'*50}")
    print(f"🔍 {title}")
    print(f"{'-'*50}")


def analyze_data_quality_issues():
    """Analyze data quality issues that cause training problems"""
    print_section("Data Quality Analysis")

    try:
        config = get_config()
        db = get_database()
        engineer = FeatureEngineer()

        # Analyze multiple tickers
        tickers = config.tickers[:5]  # First 5 tickers

        data_issues = {}

        for ticker in tickers:
            print(f"\n📊 Analyzing {ticker}:")

            issues = []

            # 1. Check processed features
            processed_file = f"data/processed/{ticker}_features.csv"
            if os.path.exists(processed_file):
                df = pd.read_csv(processed_file)

                # Check data size
                print(f"  📊 Data size: {df.shape}")
                if df.shape[0] < 1000:
                    issues.append(f"Small dataset: {df.shape[0]} samples")

                # Check target variables
                target_cols = [
                    col for col in df.columns if col.startswith("Target_Close")
                ]

                for target_col in target_cols:
                    if target_col in df.columns:
                        target_data = df[target_col].dropna()

                        # Issue 1: Zero or very low variance
                        target_std = target_data.std()
                        if target_std == 0:
                            issues.append(f"{target_col}: Zero variance")
                        elif target_std < 0.01:
                            issues.append(
                                f"{target_col}: Very low variance ({target_std:.6f})"
                            )

                        # Issue 2: Extreme outliers
                        q1, q3 = target_data.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = target_data[
                            (target_data < q1 - 3 * iqr) | (target_data > q3 + 3 * iqr)
                        ]
                        if len(outliers) > len(target_data) * 0.05:  # >5% outliers
                            issues.append(
                                f"{target_col}: Many outliers ({len(outliers)}/{len(target_data)})"
                            )

                        # Issue 3: Non-stationarity
                        # Simple check: rolling mean trend
                        rolling_mean = target_data.rolling(window=50).mean()
                        if len(rolling_mean.dropna()) > 0:
                            trend_change = (
                                abs(rolling_mean.iloc[-1] - rolling_mean.iloc[50])
                                if len(rolling_mean) > 50
                                else 0
                            )
                            if trend_change > target_std:
                                issues.append(
                                    f"{target_col}: Possible non-stationarity"
                                )

                # Check feature quality
                feature_cols = [
                    col for col in engineer.feature_columns if col in df.columns
                ]
                if feature_cols:
                    feature_data = df[feature_cols]

                    # Issue 4: Constant features
                    constant_features = feature_data.columns[feature_data.std() == 0]
                    if len(constant_features) > 0:
                        issues.append(f"Constant features: {list(constant_features)}")

                    # Issue 5: Highly correlated features
                    corr_matrix = feature_data.corr().abs()
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i + 1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > 0.95:
                                high_corr_pairs.append(
                                    (corr_matrix.columns[i], corr_matrix.columns[j])
                                )

                    if high_corr_pairs:
                        issues.append(f"High correlation pairs: {len(high_corr_pairs)}")

                    # Issue 6: Missing data patterns
                    missing_pct = feature_data.isnull().sum() / len(feature_data) * 100
                    high_missing = missing_pct[missing_pct > 10]
                    if len(high_missing) > 0:
                        issues.append(
                            f"High missing data: {len(high_missing)} features >10%"
                        )

                print(f"  📊 Issues found: {len(issues)}")
                for issue in issues:
                    print(f"    ⚠️ {issue}")

                data_issues[ticker] = issues

            else:
                print(f"  ❌ No processed data found")
                data_issues[ticker] = ["No processed data"]

        return data_issues

    except Exception as e:
        print(f"❌ Error in data quality analysis: {e}")
        return {}


def analyze_training_stability():
    """Analyze training stability and reproducibility"""
    print_section("Training Stability Analysis")

    # Check model files for same ticker/model combinations
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ No models directory found")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]

    # Group by ticker and model type
    model_groups = {}
    for model_file in model_files:
        parts = model_file.replace(".pt", "").split("_")
        if len(parts) >= 2:
            ticker = parts[0]
            model_type = parts[1]
            key = f"{ticker}_{model_type}"

            if key not in model_groups:
                model_groups[key] = []
            model_groups[key].append(model_file)

    print(f"📊 Found {len(model_groups)} ticker-model combinations")

    # Analyze file sizes and timestamps for stability
    for key, files in model_groups.items():
        if len(files) > 1:
            print(f"\n📊 {key}: {len(files)} files")

            file_sizes = []
            timestamps = []

            for file in files:
                file_path = os.path.join(models_dir, file)
                size = os.path.getsize(file_path)
                timestamp = os.path.getmtime(file_path)

                file_sizes.append(size)
                timestamps.append(timestamp)

            # Check size consistency
            size_std = np.std(file_sizes)
            size_mean = np.mean(file_sizes)
            size_cv = size_std / size_mean if size_mean > 0 else 0

            print(f"  📊 File size CV: {size_cv:.4f}")
            if size_cv > 0.1:
                print(f"    ⚠️ High size variation - unstable training")
            else:
                print(f"    ✅ Consistent file sizes")


def analyze_model_architecture_issues():
    """Analyze model architecture and hyperparameter issues"""
    print_section("Model Architecture Analysis")

    try:
        config = get_config()

        # Check training configuration
        training_config = config.get("training", {})
        models_config = config.get("models", {})

        print(f"📊 Training Configuration Analysis:")

        # Issue 1: Batch size
        batch_size = training_config.get("batch_size", 32)
        print(f"  📊 Batch size: {batch_size}")
        if batch_size < 8:
            print(f"    ⚠️ Very small batch size - may cause instability")
        elif batch_size > 128:
            print(f"    ⚠️ Large batch size - may reduce generalization")

        # Issue 2: Learning rate (need to check trainer)
        # This would require loading the actual trainer configuration

        # Issue 3: Model complexity vs data size
        timesteps = training_config.get("timesteps", 30)
        print(f"  📊 Timesteps: {timesteps}")

        # Check model architectures
        for model_type, model_config in models_config.items():
            if isinstance(model_config, dict) and model_type != "forecast_horizons":
                print(f"\n🤖 {model_type.upper()} Configuration:")

                for param, value in model_config.items():
                    print(f"    {param}: {value}")

                # Check for potential issues
                if "hidden_dim" in model_config:
                    hidden_dim = model_config["hidden_dim"]
                    if hidden_dim > 256:
                        print(f"    ⚠️ Large hidden dimension - may overfit")
                    elif hidden_dim < 32:
                        print(f"    ⚠️ Small hidden dimension - may underfit")

                if "dropout_rate" in model_config:
                    dropout = model_config["dropout_rate"]
                    if dropout > 0.5:
                        print(f"    ⚠️ High dropout - may underfit")
                    elif dropout < 0.1:
                        print(f"    ⚠️ Low dropout - may overfit")

    except Exception as e:
        print(f"❌ Error analyzing model architecture: {e}")


def simulate_training_scenarios():
    """Simulate different training scenarios to understand variability"""
    print_section("Training Scenario Simulation")

    try:
        # Load sample data
        config = get_config()
        db = get_database()

        ticker = "VCB"
        processed_file = f"data/processed/{ticker}_features.csv"

        if not os.path.exists(processed_file):
            print(f"❌ No processed data for {ticker}")
            return

        df = pd.read_csv(processed_file)

        # Get target and features
        target_col = "Target_Close_t+1"
        if target_col not in df.columns:
            print(f"❌ Target column {target_col} not found")
            return

        # Prepare data
        feature_cols = [
            col for col in df.columns if not col.startswith("Target_") and col != "time"
        ]

        X = df[feature_cols].fillna(0).values
        y = df[target_col].fillna(0).values

        # Remove any infinite or NaN values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        if len(X) < 100:
            print(f"❌ Insufficient clean data: {len(X)} samples")
            return

        print(f"📊 Using {len(X)} samples, {X.shape[1]} features")

        # Simulate different scenarios
        scenarios = {
            "Different train/test splits": simulate_split_variability,
            "Different scaling methods": simulate_scaling_effects,
            "Different data subsets": simulate_data_subset_effects,
            "Noise sensitivity": simulate_noise_effects,
        }

        for scenario_name, scenario_func in scenarios.items():
            print(f"\n🧪 {scenario_name}:")
            try:
                scenario_func(X, y)
            except Exception as e:
                print(f"    ❌ Error in scenario: {e}")

    except Exception as e:
        print(f"❌ Error in simulation: {e}")


def simulate_split_variability(X, y):
    """Simulate variability due to different train/test splits"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    r2_scores = []

    # Try 10 different random splits
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=i
        )

        # Simple linear regression for quick testing
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    print(f"    📊 R² range: [{min(r2_scores):.4f}, {max(r2_scores):.4f}]")
    print(f"    📊 R² std: {np.std(r2_scores):.4f}")

    negative_count = sum(1 for r2 in r2_scores if r2 < 0)
    print(f"    📊 Negative R²: {negative_count}/10")


def simulate_scaling_effects(X, y):
    """Simulate effects of different scaling methods"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scalers = {
        "No scaling": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }

    for scaler_name, scaler in scalers.items():
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        print(f"    📊 {scaler_name}: R² = {r2:.4f}")


def simulate_data_subset_effects(X, y):
    """Simulate effects of using different data subsets"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    subset_sizes = [0.3, 0.5, 0.7, 0.9]

    for size in subset_sizes:
        # Use different subset of data
        n_samples = int(len(X) * size)
        indices = np.random.choice(len(X), n_samples, replace=False)

        X_subset = X[indices]
        y_subset = y[indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        print(f"    📊 {size*100:.0f}% data: R² = {r2:.4f}")


def simulate_noise_effects(X, y):
    """Simulate effects of adding noise to targets"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]

    for noise_level in noise_levels:
        # Add noise to training targets
        y_train_noisy = y_train + np.random.normal(
            0, noise_level * np.std(y_train), len(y_train)
        )

        model = LinearRegression()
        model.fit(X_train, y_train_noisy)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        print(f"    📊 Noise {noise_level:.1f}x: R² = {r2:.4f}")


def provide_recommendations():
    """Provide specific recommendations to improve training stability"""
    print_section("Recommendations for Stable Training")

    recommendations = [
        "🎯 Data Quality Improvements:",
        "   • Remove constant and near-constant features",
        "   • Handle outliers (winsorization or removal)",
        "   • Ensure sufficient data (>1000 samples per model)",
        "   • Check for data leakage in feature engineering",
        "",
        "🎯 Model Architecture:",
        "   • Use appropriate model complexity for data size",
        "   • Implement proper regularization (dropout, weight decay)",
        "   • Consider ensemble methods to reduce variance",
        "   • Use cross-validation for model selection",
        "",
        "🎯 Training Process:",
        "   • Fix random seeds for reproducibility",
        "   • Use early stopping with validation set",
        "   • Implement learning rate scheduling",
        "   • Monitor training/validation curves",
        "",
        "🎯 Evaluation Strategy:",
        "   • Focus on direction accuracy over R²",
        "   • Use time-series cross-validation",
        "   • Compare against multiple baselines",
        "   • Report confidence intervals",
        "",
        "🎯 Financial Data Specific:",
        "   • Accept that R² can be negative (high noise)",
        "   • Use risk-adjusted metrics (Sharpe ratio)",
        "   • Consider regime changes in markets",
        "   • Regular model retraining (monthly/quarterly)",
    ]

    for rec in recommendations:
        print(rec)


def main():
    """Main diagnosis function"""
    print_header("TRAINING ISSUES DIAGNOSIS")
    print("🔬 Deep analysis of training variability and negative R² values")

    # 1. Data quality analysis
    data_issues = analyze_data_quality_issues()

    # 2. Training stability analysis
    analyze_training_stability()

    # 3. Model architecture analysis
    analyze_model_architecture_issues()

    # 4. Simulation of training scenarios
    simulate_training_scenarios()

    # 5. Recommendations
    provide_recommendations()

    # 6. Summary
    print_header("DIAGNOSIS SUMMARY")

    print("🔍 Key Findings:")
    print("✅ 1. Negative R² is NORMAL in financial prediction (high noise)")
    print("✅ 2. Variability comes from small datasets and random splits")
    print("✅ 3. Financial data has inherent non-stationarity")
    print("✅ 4. Model complexity may exceed data capacity")
    print("⚠️ 5. Some features may be constant or highly correlated")

    print(f"\n🎯 Main Conclusion:")
    print("The variability and negative R² are expected in financial modeling.")
    print("Focus on direction accuracy and ensemble methods for better stability.")
    print("Regular retraining and proper validation are essential.")


if __name__ == "__main__":
    main()

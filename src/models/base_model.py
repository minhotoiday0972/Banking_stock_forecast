# src/models/base_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger("models")

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BaseModel(nn.Module, ABC):
    """Base class for all models"""
    
    def __init__(self, input_dim: int, config: Dict[str, Any]):
        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.config = config
        self.horizons = config.get('forecast_horizons', [1, 3, 5])
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass - must return dict with keys for each target"""
        pass

class ModelTrainer:
    """Centralized model training"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def _calculate_class_weights(self, target_data: np.ndarray) -> torch.Tensor:
        """Calculate enhanced class weights for imbalanced classification"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get unique classes and their weights
        unique_classes = np.unique(target_data)
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_classes, 
            y=target_data
        )
        
        # Convert to tensor
        weights_tensor = torch.zeros(3)  # 3 classes: Down(0), Flat(1), Up(2)
        for i, class_idx in enumerate(unique_classes):
            weights_tensor[int(class_idx)] = class_weights[i]
        
        # ENHANCED: Apply EXTREME penalty to dominant class
        # Find dominant class (usually class 1 - Flat)
        class_counts = np.bincount(target_data.astype(int), minlength=3)
        dominant_class = np.argmax(class_counts)
        
        # EXTREME enhancement for stubborn models
        enhancement_factor = 10.0  # Much stronger
        for i in range(3):
            if i == dominant_class:
                weights_tensor[i] = weights_tensor[i] * 0.1  # EXTREME penalty for dominant
            else:
                weights_tensor[i] = weights_tensor[i] * enhancement_factor  # EXTREME boost minorities
        
        logger.info(f"Class distribution: {class_counts}")
        logger.info(f"Dominant class: {dominant_class}")
        logger.info(f"Enhanced class weights: {weights_tensor.tolist()}")
        return weights_tensor
    
    def prepare_data(self, sequences: Dict[str, np.ndarray], 
                    train_split: float = 0.8, val_split: float = 0.1) -> Dict[str, Any]:
        """Prepare data for training"""
        X = sequences['X']
        total_samples = len(X)
        
        # Calculate split indices
        train_size = int(total_samples * train_split)
        val_size = int(total_samples * val_split)
        
        # Split data
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        # Prepare targets
        targets = {}
        for key, values in sequences.items():
            if key.startswith('Target_'):
                targets[key] = {
                    'train': values[:train_size],
                    'val': values[train_size:train_size + val_size],
                    'test': values[train_size + val_size:]
                }
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'targets': targets,
            'input_dim': X.shape[2]
        }
    
    def create_data_loaders(self, X: np.ndarray, targets: Dict[str, np.ndarray], 
                           batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Create data loader"""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        target_tensors = []
        
        for key in sorted(targets.keys()):
            target_tensors.append(torch.tensor(targets[key], dtype=torch.float32))
        
        dataset = TensorDataset(X_tensor, *target_tensors)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def calculate_metrics(self, predictions: Dict[str, np.ndarray], 
                         targets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive metrics for all targets"""
        metrics = {}
        
        for target_name in predictions.keys():
            if target_name not in targets:
                continue
                
            pred = predictions[target_name]
            true = targets[target_name]
            
            if 'Direction' in target_name:
                # Enhanced classification metrics
                accuracy = accuracy_score(true, pred)
                
                # Per-class metrics
                precision, recall, f1, support = precision_recall_fscore_support(
                    true, pred, average=None, zero_division=0
                )
                
                # Balanced accuracy (more reliable for imbalanced data)
                balanced_acc = accuracy_score(true, pred)  # Will implement balanced version
                
                # Confusion matrix for detailed analysis
                cm = confusion_matrix(true, pred, labels=[0, 1, 2])
                
                # Check if model is just predicting one class
                unique_predictions = len(np.unique(pred))
                is_constant_prediction = unique_predictions == 1
                
                metrics[target_name] = {
                    'accuracy': accuracy,
                    'balanced_accuracy': balanced_acc,
                    'precision_per_class': precision.tolist(),
                    'recall_per_class': recall.tolist(),
                    'f1_per_class': f1.tolist(),
                    'support_per_class': support.tolist(),
                    'confusion_matrix': cm.tolist(),
                    'unique_predictions': unique_predictions,
                    'is_constant_prediction': is_constant_prediction,
                    'type': 'classification'
                }
                
                # Log warning if constant prediction detected
                if is_constant_prediction:
                    predicted_class = pred[0]
                    logger.warning(f"⚠️  {target_name}: Model predicting ONLY class {predicted_class}! (Fake learning detected)")
                    logger.warning(f"   Accuracy {accuracy:.4f} is MISLEADING!")
                
            else:
                # Regression metrics
                mse = mean_squared_error(true, pred)
                mae = mean_absolute_error(true, pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(true, pred)
                
                metrics[target_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'type': 'regression'
                }
        
        return metrics
    
    def train_model(self, model: BaseModel, data: Dict[str, Any], 
                   epochs: int = 50, batch_size: int = 32, 
                   learning_rate: float = 0.001, 
                   early_stopping_patience: int = 10) -> Dict[str, Any]:
        """Train model with early stopping"""
        
        model = model.to(self.device)
        
        # Prepare optimizers and criteria
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create data loaders
        train_targets = {k: v['train'] for k, v in data['targets'].items()}
        val_targets = {k: v['val'] for k, v in data['targets'].items()}
        test_targets = {k: v['test'] for k, v in data['targets'].items()}
        
        train_loader = self.create_data_loaders(data['X_train'], train_targets, batch_size, True)
        val_loader = self.create_data_loaders(data['X_val'], val_targets, batch_size, False)
        test_loader = self.create_data_loaders(data['X_test'], test_targets, batch_size, False)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_data in train_loader:
                X_batch = batch_data[0].to(self.device)
                target_batch = [t.to(self.device) for t in batch_data[1:]]
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(X_batch)
                
                # Calculate loss for all targets with enhanced weighting
                total_loss = 0
                regression_loss = 0
                classification_loss = 0
                target_keys = sorted(data['targets'].keys())
                
                for i, target_key in enumerate(target_keys):
                    if 'Direction' in target_key:
                        # ENHANCED: Use Focal Loss instead of CrossEntropy
                        if not hasattr(self, 'class_weights'):
                            self.class_weights = self._calculate_class_weights(data['targets'][target_key]['train'])
                        
                        # Use EXTREME Focal Loss for stubborn models
                        focal_criterion = FocalLoss(alpha=2.0, gamma=5.0)  # Much stronger
                        loss = focal_criterion(outputs[target_key], target_batch[i].long())
                        
                        # Apply class weights manually to focal loss
                        class_weights_expanded = self.class_weights.to(self.device)[target_batch[i].long()]
                        loss = (loss * class_weights_expanded).mean()
                        
                        classification_loss += loss
                    else:
                        # Regression loss
                        criterion = nn.MSELoss()
                        loss = criterion(outputs[target_key].squeeze(), target_batch[i])
                        regression_loss += loss
                
                # EXTREME: Weighted combination with HEAVY emphasis on classification
                classification_weight = 5.0  # EXTREME emphasis
                regression_weight = 1.0
                
                total_loss = (regression_weight * regression_loss + 
                             classification_weight * classification_loss)
                
                total_loss.backward()
                optimizer.step()
                train_loss += total_loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_predictions = {key: [] for key in data['targets'].keys()}
            val_true = {key: [] for key in data['targets'].keys()}
            
            with torch.no_grad():
                for batch_data in val_loader:
                    X_batch = batch_data[0].to(self.device)
                    target_batch = [t.to(self.device) for t in batch_data[1:]]
                    
                    outputs = model(X_batch)
                    
                    # Calculate validation loss and collect predictions
                    total_loss = 0
                    target_keys = sorted(data['targets'].keys())
                    
                    for i, target_key in enumerate(target_keys):
                        if 'Direction' in target_key:
                            # Use EXTREME Focal Loss for validation too
                            focal_criterion = FocalLoss(alpha=2.0, gamma=5.0)  # Much stronger
                            loss = focal_criterion(outputs[target_key], target_batch[i].long())
                            
                            # Apply class weights
                            class_weights_expanded = self.class_weights.to(self.device)[target_batch[i].long()]
                            loss = (loss * class_weights_expanded).mean()
                            
                            # Get predicted classes
                            pred_classes = torch.argmax(outputs[target_key], dim=1)
                            val_predictions[target_key].extend(pred_classes.cpu().numpy())
                            
                            total_loss += loss * 5.0  # Same EXTREME weighting as training
                        else:
                            criterion = nn.MSELoss()
                            loss = criterion(outputs[target_key].squeeze(), target_batch[i])
                            val_predictions[target_key].extend(outputs[target_key].squeeze().cpu().numpy())
                            total_loss += loss
                        
                        val_true[target_key].extend(target_batch[i].cpu().numpy())
                    
                    val_loss += total_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # Calculate metrics
            val_predictions_np = {k: np.array(v) for k, v in val_predictions.items()}
            val_true_np = {k: np.array(v) for k, v in val_true.items()}
            metrics = self.calculate_metrics(val_predictions_np, val_true_np)
            history['metrics'].append(metrics)
            
            # Enhanced Logging
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
                
                # Log detailed metrics
                for target_name, target_metrics in metrics.items():
                    if target_metrics['type'] == 'regression':
                        logger.info(f"  {target_name} - RMSE: {target_metrics['rmse']:.6f}, R²: {target_metrics['r2']:.6f}")
                    else:
                        # Enhanced classification logging
                        acc = target_metrics['accuracy']
                        is_constant = target_metrics['is_constant_prediction']
                        unique_preds = target_metrics['unique_predictions']
                        
                        if is_constant:
                            logger.warning(f"  🚨 {target_name} - Accuracy: {acc:.6f} [FAKE - Only {unique_preds} class predicted!]")
                        else:
                            logger.info(f"  ✅ {target_name} - Accuracy: {acc:.6f} [Real - {unique_preds} classes predicted]")
                        
                        # Log per-class metrics
                        precision = target_metrics['precision_per_class']
                        recall = target_metrics['recall_per_class']
                        f1 = target_metrics['f1_per_class']
                        
                        logger.info(f"    Per-class Precision: Down={precision[0]:.3f}, Flat={precision[1]:.3f}, Up={precision[2]:.3f}")
                        logger.info(f"    Per-class Recall:    Down={recall[0]:.3f}, Flat={recall[1]:.3f}, Up={recall[2]:.3f}")
                        logger.info(f"    Per-class F1:        Down={f1[0]:.3f}, Flat={f1[1]:.3f}, Up={f1[2]:.3f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Test evaluation
        test_metrics = self._evaluate_model(model, test_loader, data['targets'])
        
        return {
            'model': model,
            'history': history,
            'test_metrics': test_metrics,
            'best_val_loss': best_val_loss
        }
    
    def _evaluate_model(self, model: BaseModel, test_loader: DataLoader, 
                       targets_info: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate model on test set"""
        model.eval()
        test_predictions = {key: [] for key in targets_info.keys()}
        test_true = {key: [] for key in targets_info.keys()}
        
        with torch.no_grad():
            for batch_data in test_loader:
                X_batch = batch_data[0].to(self.device)
                target_batch = [t.to(self.device) for t in batch_data[1:]]
                
                outputs = model(X_batch)
                target_keys = sorted(targets_info.keys())
                
                for i, target_key in enumerate(target_keys):
                    if 'Direction' in target_key:
                        pred_classes = torch.argmax(outputs[target_key], dim=1)
                        test_predictions[target_key].extend(pred_classes.cpu().numpy())
                    else:
                        test_predictions[target_key].extend(outputs[target_key].squeeze().cpu().numpy())
                    
                    test_true[target_key].extend(target_batch[i].cpu().numpy())
        
        # Calculate test metrics
        test_predictions_np = {k: np.array(v) for k, v in test_predictions.items()}
        test_true_np = {k: np.array(v) for k, v in test_true.items()}
        
        return self.calculate_metrics(test_predictions_np, test_true_np)
    
    def save_model(self, model: BaseModel, ticker: str, model_type: str):
        """Save trained model"""
        models_dir = self.config.models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"{ticker}_{model_type}_best.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model: {model_path}")
    
    def plot_training_history(self, history: Dict[str, List], ticker: str, model_type: str):
        """Plot training history"""
        outputs_dir = self.config.outputs_dir
        os.makedirs(outputs_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{ticker} - {model_type} Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Metrics plot (example for first regression target)
        plt.subplot(1, 2, 2)
        if history['metrics']:
            # Find first regression target
            regression_target = None
            for target_name in history['metrics'][0].keys():
                if history['metrics'][0][target_name]['type'] == 'regression':
                    regression_target = target_name
                    break
            
            if regression_target:
                r2_scores = [m[regression_target]['r2'] for m in history['metrics']]
                plt.plot(r2_scores, label=f'{regression_target} R²')
                plt.title(f'{ticker} - {model_type} R² Score')
                plt.xlabel('Epoch')
                plt.ylabel('R² Score')
                plt.legend()
                plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(outputs_dir, f"{ticker}_{model_type}_training.png")
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved training plot: {plot_path}")
"""
Model training module for generic image classification.
Handles model creation, training loop, and checkpointing.
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping to avoid overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class ModelTrainer:
    """
    Handles model training, validation, and checkpointing.
    """
    
    def __init__(self, model: nn.Module, config, class_weights: torch.Tensor = None):
        """
        Args:
            model: PyTorch model to train
            config: Configuration dictionary
            device: Device to use (cuda or cpu)
            class_weights: Optional class weights for loss function
        """
        self.model = model
        self.config = config

        # Important Improvement
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        early_stopping_config = config.TRAINING_CONFIG
        self.early_stopping = EarlyStopping(
            patience=early_stopping_config['early_stopping_patience'],
            min_delta=early_stopping_config['early_stopping_min_delta']
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config."""
        training_config = self.config.TRAINING_CONFIG
        optimizer_name = training_config['optimizer'].lower()
        lr = training_config['learning_rate']
        weight_decay = training_config['weight_decay']
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, 
                           weight_decay=weight_decay)
        else:
            print(f"⚠ Unknown optimizer {optimizer_name}, using Adam")
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler based on config."""
        training_config = self.config.TRAINING_CONFIG
        scheduler_name = training_config['scheduler'].lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs']
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config['num_epochs'] // 3,
                gamma=0.1
            )
        elif scheduler_name == 'exponential':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        else:
            print(f"⚠ Unknown scheduler {scheduler_name}, no scheduler used")
            return None
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            train_loss, train_accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            val_loss, val_accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating", leave=False)
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded from {path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Complete training loop.
        """
        num_epochs = self.config.TRAINING_CONFIG['num_epochs']
        best_val_acc = 0

        print(f"\n✍ Starting training for {num_epochs} epochs...")
        print(f"✓ Training device: {self.device}")
        print(f"✓ Model is on: {next(self.model.parameters()).device}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Log
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(self.config.CHECKPOINT_CONFIG['best_model_path'], 
                                   is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss):
                print(
                    f"⚠ Early stopping triggered after {epoch+1} epochs"
                )
                break
        
        # Save final model
        self.save_checkpoint(self.config.CHECKPOINT_CONFIG['final_model_path'])
        
        print("\n✓ Training completed!")
        
        return self.history

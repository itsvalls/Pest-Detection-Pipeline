"""
Model building module for crop classification.
Creates and loads pre-trained models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple


class ModelBuilder:
    """
    Builds different model architectures for crop classification.
    Supports transfer learning with pretrained weights.
    """
    
    @staticmethod
    def build_model(num_classes: int, architecture: str = 'resnet50', 
                   pretrained: bool = True, dropout_rate: float = 0.3) -> nn.Module:
        """
        Build classification model based on specified architecture.
        
        Args:
            num_classes: Number of output classes
            architecture: Model architecture name (resnet50, resnet18, mobilenet_v2, efficientnet_b0)
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Dropout rate for regularization
            
        Returns:
            PyTorch model
        """
        try:
            import torchvision.models as models
        except ImportError:
            print("✗ torchvision not installed. Install it with: pip install torchvision")
            raise
        
        architecture = architecture.lower()
        model = None
        num_features = 0
        
        # Helper to get weights if available
        def get_model(name, pretrained_flag):
            # Map architecture names to their weights enum class names
            weight_enums = {
                'resnet50': 'ResNet50_Weights',
                'resnet18': 'ResNet18_Weights',
                'mobilenet_v2': 'MobileNet_V2_Weights',
                'efficientnet_b0': 'EfficientNet_B0_Weights'
            }
            
            # Check if using new weights API
            if hasattr(models, weight_enums.get(name, '')):
                if pretrained_flag:
                    weights = getattr(models, weight_enums[name]).DEFAULT
                    return getattr(models, name)(weights=weights)
                else:
                    return getattr(models, name)(weights=None)
            else:
                # Fallback to old API
                return getattr(models, name)(pretrained=pretrained_flag)

        if architecture == 'resnet50':
            model = get_model('resnet50', pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        elif architecture == 'resnet18':
            model = get_model('resnet18', pretrained)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        elif architecture == 'mobilenet_v2':
            model = get_model('mobilenet_v2', pretrained)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        elif architecture == 'efficientnet_b0':
            model = get_model('efficientnet_b0', pretrained)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        print(f"✓ {architecture} model built with {num_classes} classes (pretrained={pretrained})")
        
        return model
    
    @staticmethod
    def load_pretrained_weights(model: nn.Module, checkpoint_path: Path, 
                               device: str = 'cuda'):
        """
        Load pretrained weights into model.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
            device: Device (cuda or cpu)
        """
        if not checkpoint_path.exists():
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
            return model
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded pretrained weights from {checkpoint_path}")
        
        return model
    
    @staticmethod
    def freeze_backbone(model: nn.Module):
        """
        Freeze backbone weights for transfer learning, leaving only the final
        classification layer trainable.
        
        Args:
            model: Model to freeze
        """
        # Freeze all parameters initially
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the final classification layer
        # For ResNet-like models, the final layer is typically 'fc'
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Module):
            for param in model.fc.parameters():
                param.requires_grad = True
            print("✓ Froze backbone, unfroze 'fc' layer for training.")
        # For MobileNet/EfficientNet-like models, the final layer is typically 'classifier'
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            for param in model.classifier.parameters():
                param.requires_grad = True
            print("✓ Froze backbone, unfroze 'classifier' layer for training.")
        else:
            print("⚠ Warning: Could not identify a standard classification layer (fc or classifier) to unfreeze.")

        ModelBuilder.count_parameters(model)
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Tuple[int, int]:
        """
        Count trainable and total parameters.
        
        Returns:
            (trainable_params, total_params)
        """
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model parameters - Total: {total:,}, Trainable: {trainable:,}")
        
        return trainable, total

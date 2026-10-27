import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        score = output[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            return None
            
        b, k, u, v = gradients.size()
        
        # Global Average Pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.view(u, v).detach().cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        # We need to hold references to the hooks to remove them later if needed, 
        # but for this app we can keep them.
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple (tensor,)
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM++ heatmap
        Args:
            x: Input tensor (1, C, H, W)
            class_idx: Target class index. If None, uses the highest probability class.
        Returns:
            heatmap: Numpy array of shape (H_feature, W_feature)
        """
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[:, class_idx]
        
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass
        score.backward(retain_graph=True)
        
        gradients = self.gradients # [B, C, H, W]
        activations = self.activations # [B, C, H, W]
        
        if gradients is None or activations is None:
            return None
        
        b, k, u, v = gradients.size()
        
        # Grad-CAM++ calculation
        # Alpha numerator: gradients^2
        alpha_num = gradients.pow(2)
        
        # Alpha denominator: 2*gradients^2 + sum(activations * gradients^3)
        # Sum over spatial dimensions (2, 3)
        alpha_denom = 2 * gradients.pow(2) + \
            torch.sum(activations * gradients.pow(3), dim=(2, 3), keepdim=True)
            
        # Avoid division by zero with safe mask
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        alphas = alpha_num / alpha_denom
        
        # Weights = sum(alphas * relu(gradients))
        # Only positive gradients contribute
        weights = torch.max(gradients, torch.zeros_like(gradients))
        weights = torch.sum(alphas * weights, dim=(2, 3), keepdim=True)
        
        # CAM = sum(weights * activations)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Extract the single map
        cam = cam.view(u, v).detach().cpu().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam

def overlay_heatmap(image_pil, cam_mask, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image
    """
    # Convert PIL to numpy (RGB)
    img = np.array(image_pil)
    
    # Resize CAM to image size
    heatmap = cv2.resize(cam_mask, (img.shape[1], img.shape[0]))
    
    # Convert to uint8 and apply colormap
    heatmap = np.uint8(255 * heatmap)
    
    # cv2.applyColorMap expects BGR image or grayscale, returns BGR
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    
    # Convert heatmap from BGR to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose
    superimposed_img = heatmap_colored * alpha + img * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    return Image.fromarray(superimposed_img)

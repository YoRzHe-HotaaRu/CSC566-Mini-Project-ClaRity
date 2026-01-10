"""
Deep Learning Module for Road Surface Layer Analyzer
Implements DeepLabv3+ semantic segmentation with CUDA support.

CSC566 Image Processing Mini Project
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

from .config import DEEP_LEARNING_CONFIG, ROAD_LAYERS, LAYER_COLORS_RGB


class RoadLayerDataset(Dataset):
    """
    PyTorch Dataset for road layer images.
    """
    
    def __init__(
        self,
        image_paths: list,
        mask_paths: Optional[list] = None,
        transform=None,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths (for training)
            transform: Optional transforms
            target_size: Target image size (H, W)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.mask_paths is not None:
            # Load mask
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask = torch.from_numpy(mask).long()
            return image, mask
        
        return image


class DeepLabSegmenter:
    """
    DeepLabv3+ semantic segmentation for road layers.
    Uses CUDA GPU acceleration when available.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        encoder_name: str = None,
        encoder_weights: str = None,
        n_classes: int = None,
        use_cuda: bool = True
    ):
        """
        Initialize DeepLabv3+ segmenter.
        
        Args:
            model_path: Path to pre-trained weights (optional)
            encoder_name: Backbone encoder (e.g., "resnet101")
            encoder_weights: Pre-trained weights for encoder
            n_classes: Number of output classes
            use_cuda: Whether to use CUDA GPU
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning. Install with: pip install torch torchvision")
        
        if not SMP_AVAILABLE:
            raise ImportError("segmentation-models-pytorch is required. Install with: pip install segmentation-models-pytorch")
        
        # Configuration
        config = DEEP_LEARNING_CONFIG
        self.encoder_name = encoder_name or config["encoder_name"]
        self.encoder_weights = encoder_weights or config["encoder_weights"]
        self.n_classes = n_classes or config["classes"]
        self.image_size = config["image_size"]
        
        # Device selection
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        # Create model
        self.model = smp.DeepLabV3Plus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=3,
            classes=self.n_classes
        )
        
        # Load pre-trained weights if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform for inference
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.image_size)
        
        # Apply transforms
        tensor = self.transform(image_resized)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def segment(
        self,
        image: np.ndarray,
        return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform semantic segmentation.
        
        Args:
            image: Input image
            return_proba: Whether to return probability maps
            
        Returns:
            Segmentation labels (and optionally probabilities)
        """
        # Store original size
        original_size = (image.shape[1], image.shape[0])
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if return_proba:
                proba = torch.softmax(output, dim=1)
                proba = proba.cpu().numpy()[0]
            
            # Get class predictions
            predictions = torch.argmax(output, dim=1)
            labels = predictions.cpu().numpy()[0]
        
        # Resize back to original size
        labels = cv2.resize(
            labels.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convert 0-indexed to 1-indexed (to match ROAD_LAYERS)
        labels = labels + 1
        
        if return_proba:
            # Resize probability maps
            proba_resized = np.zeros((self.n_classes, original_size[1], original_size[0]))
            for i in range(self.n_classes):
                proba_resized[i] = cv2.resize(proba[i], original_size)
            return labels, proba_resized
        
        return labels
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Fine-tune model on road layer dataset.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        self.model.train()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            self.model.train()
            
            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                
                scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    self.save_model(save_path)
                
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f} - "
                      f"Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        self.model.eval()
        return history
    
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (validation loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == masks).sum().item()
                total += masks.numel()
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        return val_loss, accuracy
    
    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(masks.numpy())
        
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        
        # Compute metrics
        accuracy = (predictions == labels).mean()
        
        # IoU per class
        iou_per_class = {}
        for c in range(self.n_classes):
            intersection = ((predictions == c) & (labels == c)).sum()
            union = ((predictions == c) | (labels == c)).sum()
            iou = intersection / (union + 1e-10)
            iou_per_class[ROAD_LAYERS[c+1]["name"]] = float(iou)
        
        mean_iou = np.mean(list(iou_per_class.values()))
        
        return {
            "accuracy": float(accuracy),
            "mean_iou": float(mean_iou),
            "iou_per_class": iou_per_class
        }
    
    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    
    def create_colored_output(
        self,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Create colored visualization of segmentation.
        
        Args:
            labels: Segmentation labels (1-indexed)
            
        Returns:
            RGB colored image
        """
        h, w = labels.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for layer_num in range(1, 6):
            mask = labels == layer_num
            if mask.any():
                color = ROAD_LAYERS[layer_num]["color"]
                colored[mask] = color
        
        return colored


def check_cuda_available() -> Dict:
    """
    Check CUDA availability and GPU information.
    
    Returns:
        Dictionary with CUDA info
    """
    if not TORCH_AVAILABLE:
        return {"available": False, "reason": "PyTorch not installed"}
    
    if torch.cuda.is_available():
        return {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_cached": torch.cuda.memory_reserved(0)
        }
    
    return {"available": False, "reason": "CUDA not available"}

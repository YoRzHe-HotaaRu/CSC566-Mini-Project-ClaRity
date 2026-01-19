"""
YOLOv11 Instance Segmentation Analyzer
Handles model loading, inference, and mask visualization for road layers.

CSC566 Image Processing Mini Project
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO


class YOLOAnalyzer:
    """YOLOv11 instance segmentation analyzer for road layers."""
    
    # Class name to layer number mapping (matching existing ROAD_LAYERS)
    CLASS_TO_LAYER = {
        'layer_subgrade': 1,
        'layer_subbase': 2,
        'layer_basecourse': 3,
        'layer_bindercourse': 4,
        'layer_surfacecourse': 5
    }
    
    # Layer colors (BGR format, matching config.py ROAD_LAYERS)
    LAYER_COLORS = {
        1: (43, 90, 139),    # Brown - Subgrade
        2: (128, 128, 128),  # Gray - Subbase
        3: (169, 169, 169),  # Light Gray - Base
        4: (70, 70, 70),     # Dark Gray - Binder
        5: (50, 50, 50)      # Very Dark Gray - Surface
    }
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize YOLOv11 analyzer.
        
        Args:
            model_path: Path to .pt weights file
            device: 'cuda' or 'cpu'
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent / "YOLOv11" / "YOLOv11_11_weight.pt"
        
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.confidence = 0.5
        self.iou_threshold = 0.45
        
    def load_model(self) -> bool:
        """Load the YOLO model."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.model = YOLO(str(self.model_path))
            self.model.to(self.device)
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False
    
    def set_confidence(self, conf: float):
        """Set confidence threshold (0.0 - 1.0)."""
        self.confidence = max(0.0, min(1.0, conf))
    
    def set_iou_threshold(self, iou: float):
        """Set IoU threshold (0.0 - 1.0)."""
        self.iou_threshold = max(0.0, min(1.0, iou))
    
    def preprocess(self, image: np.ndarray, sharpen: bool = False, 
                   edge_detection: bool = False, 
                   contrast_enhance: bool = False) -> np.ndarray:
        """
        Apply preprocessing to image.
        
        Args:
            image: Input BGR image
            sharpen: Apply sharpening
            edge_detection: Add edge overlay
            contrast_enhance: Apply CLAHE contrast enhancement
            
        Returns:
            Preprocessed image
        """
        result = image.copy()
        
        if contrast_enhance:
            # Apply CLAHE to each channel
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        if sharpen:
            kernel = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
            result = cv2.filter2D(result, -1, kernel)
        
        if edge_detection:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(result, 0.8, edges_colored, 0.2, 0)
        
        return result
    
    def predict(self, image: np.ndarray, preprocess_opts: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run instance segmentation on image.
        
        Args:
            image: Input BGR image
            preprocess_opts: Dict with sharpen, edge_detection, contrast_enhance
            
        Returns:
            Dict with masks, boxes, labels, confidences, visualization
        """
        if self.model is None:
            if not self.load_model():
                return {"error": "Failed to load model"}
        
        # Apply preprocessing
        if preprocess_opts:
            image = self.preprocess(
                image,
                sharpen=preprocess_opts.get("sharpen", False),
                edge_detection=preprocess_opts.get("edge_detection", False),
                contrast_enhance=preprocess_opts.get("contrast_enhance", False)
            )
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        result = results[0]
        
        # Extract detection info
        detections = []
        masks_combined = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if result.masks is not None and len(result.masks) > 0:
            for i, (box, mask, cls, conf) in enumerate(zip(
                result.boxes.xyxy.cpu().numpy(),
                result.masks.data.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy()
            )):
                class_name = result.names[int(cls)]
                layer_num = self.CLASS_TO_LAYER.get(class_name, 0)
                
                # Resize mask to original image size
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Add to combined mask
                masks_combined[mask_binary > 0] = layer_num
                
                detections.append({
                    "class_name": class_name,
                    "layer_number": layer_num,
                    "confidence": float(conf),
                    "bbox": box.tolist(),
                    "mask": mask_binary
                })
        
        # Determine dominant layer
        dominant_layer = 0
        dominant_confidence = 0.0
        if detections:
            # Find detection with largest area
            areas = [(d["mask"].sum(), d) for d in detections]
            areas.sort(key=lambda x: x[0], reverse=True)
            if areas:
                dominant = areas[0][1]
                dominant_layer = dominant["layer_number"]
                dominant_confidence = dominant["confidence"]
        
        return {
            "detections": detections,
            "masks": masks_combined,
            "dominant_layer": dominant_layer,
            "dominant_confidence": dominant_confidence,
            "preprocessed": image
        }
    
    def visualize(self, image: np.ndarray, prediction: Dict, 
                  show_masks: bool = True, show_labels: bool = True,
                  show_confidence: bool = True, mask_opacity: float = 0.4) -> np.ndarray:
        """
        Create visualization with mask overlays.
        
        Args:
            image: Original BGR image
            prediction: Result from predict()
            show_masks: Draw segmentation masks
            show_labels: Draw class labels
            show_confidence: Show confidence scores
            mask_opacity: Transparency of mask overlay (0.0 - 1.0)
            
        Returns:
            Visualization image
        """
        vis = image.copy()
        
        if show_masks and "masks" in prediction:
            masks = prediction["masks"]
            overlay = vis.copy()
            
            for layer_num in range(1, 6):
                mask = (masks == layer_num)
                if mask.any():
                    color = self.LAYER_COLORS.get(layer_num, (128, 128, 128))
                    overlay[mask] = color
            
            vis = cv2.addWeighted(vis, 1 - mask_opacity, overlay, mask_opacity, 0)
        
        # Draw bounding boxes and labels
        for det in prediction.get("detections", []):
            if show_labels or show_confidence:
                x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
                layer_num = det["layer_number"]
                color = self.LAYER_COLORS.get(layer_num, (128, 128, 128))
                
                # Draw box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_parts = []
                if show_labels:
                    from src.config import ROAD_LAYERS
                    layer_name = ROAD_LAYERS.get(layer_num, {}).get("name", det["class_name"])
                    label_parts.append(f"L{layer_num}: {layer_name}")
                if show_confidence:
                    label_parts.append(f"{det['confidence']:.0%}")
                
                label = " | ".join(label_parts)
                
                # Background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(vis, label, (x1 + 2, y1 - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis


def test_yolo_analyzer():
    """Quick test of the YOLO analyzer."""
    analyzer = YOLOAnalyzer()
    print("Loading model...")
    if analyzer.load_model():
        print("✅ Model loaded successfully!")
        print(f"   Device: {analyzer.device}")
        print(f"   Classes: {list(analyzer.CLASS_TO_LAYER.keys())}")
        return True
    else:
        print("❌ Failed to load model")
        return False


if __name__ == "__main__":
    test_yolo_analyzer()

"""
Classical Analysis Result Dialogs
Shows intermediate processing steps for Image Segmentation and Texture Feature Extraction.

CSC566 Image Processing Mini Project
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QScrollArea, QWidget, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


def numpy_to_qpixmap(image: np.ndarray, target_size: Tuple[int, int] = None) -> QPixmap:
    """Convert numpy array to QPixmap, optionally resized to target_size."""
    if image is None or image.size == 0:
        return QPixmap()
    
    # Make a copy to avoid modifying original
    img = image.copy()
    
    # Convert grayscale to BGR for display
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Resize if target size specified
    if target_size:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 0 and new_h > 0:
            img = cv2.resize(img, (new_w, new_h))
    
    # Convert BGR to RGB for Qt
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ScalableImageLabel(QLabel):
    """A QLabel that stores original image and scales it on resize."""
    
    def __init__(self, image: np.ndarray = None, parent=None):
        super().__init__(parent)
        self.original_image = image
        self.setMinimumSize(100, 100)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a2e; border: 1px solid #444;")
        if image is not None:
            self._update_pixmap()
    
    def setImage(self, image: np.ndarray):
        """Set the image to display."""
        self.original_image = image
        self._update_pixmap()
    
    def _update_pixmap(self):
        """Update the pixmap based on current size."""
        if self.original_image is None:
            return
        size = self.size()
        pixmap = numpy_to_qpixmap(self.original_image, (size.width() - 10, size.height() - 10))
        self.setPixmap(pixmap)
    
    def resizeEvent(self, event):
        """Handle resize by updating pixmap."""
        super().resizeEvent(event)
        self._update_pixmap()


class ImageSegmentationResultsDialog(QDialog):
    """
    Dialog showing intermediate Image Segmentation steps.
    Shows 6 images: Original, Sobel Edge, Dilated Gradient, Filled Holes,
    Erosion Mask, and Segmented Image.
    """
    
    def __init__(self, images: Dict[str, np.ndarray], parent=None):
        """
        Initialize dialog.
        
        Args:
            images: Dictionary with keys:
                - 'original': Original image
                - 'sobel_edge': Sobel edge detection result
                - 'dilated_gradient': Dilated gradient mask
                - 'filled_holes': Filled holes and cleared border
                - 'erosion_mask': Erosion gradient mask with small regions removed
                - 'segmented': Final segmented image
        """
        super().__init__(parent)
        self.images = images
        self.setWindowTitle("Image Segmentation Results")
        self.setMinimumSize(900, 500)
        # Add minimize/maximize buttons
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI layout - 3x3 grid with scalable images."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Table 1: Sample of Image Segmentation Results")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Grid for images - 3x3 layout (2 rows of images)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(10)
        
        # Image data: (key, label, row, col)
        image_data = [
            ('original', "Original Image", 0, 0),
            ('sobel_edge', "Sobel Edge Detection", 0, 1),
            ('dilated_gradient', "Dilated Gradient Mask", 0, 2),
            ('filled_holes', "Filled In Holes And\nCleared Border Image", 1, 0),
            ('erosion_mask', "Erosion Gradient Mask\nAnd Remove Small Region", 1, 1),
            ('segmented', "Segmented Image", 1, 2),
        ]
        
        for key, label_text, row, col in image_data:
            # Container for label + image
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(5)
            container_layout.setContentsMargins(5, 5, 5, 5)
            
            # Label
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
            lbl.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(lbl)
            
            # Image - use ScalableImageLabel for resize support
            img = self.images.get(key)
            img_lbl = ScalableImageLabel(img)
            container_layout.addWidget(img_lbl, 1)  # stretch factor 1
            
            grid.addWidget(container, row, col)
        
        layout.addWidget(grid_widget, 1)  # stretch factor 1
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("padding: 8px 20px;")
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: white; }
            QLabel { color: white; }
            QPushButton { 
                background-color: #4a4a5a; 
                color: white; 
                border: none; 
                border-radius: 5px; 
            }
            QPushButton:hover { background-color: #5a5a6a; }
        """)


class TextureFeatureResultsDialog(QDialog):
    """
    Dialog showing intermediate Texture Feature Extraction steps.
    Shows 5 images + statistical results: Original, Binarization, Segmented,
    Grayscale, Region of Interest, and Mean/StdDev/Smoothness values.
    """
    
    def __init__(self, images: Dict[str, np.ndarray], stats: Dict[str, float], parent=None):
        """
        Initialize dialog.
        
        Args:
            images: Dictionary with keys:
                - 'original': Original image
                - 'binarization': Binary threshold image
                - 'segmented': Segmented image
                - 'grayscale': Grayscale image
                - 'roi': Region of interest
            stats: Dictionary with keys:
                - 'mean': Mean value
                - 'std': Standard deviation
                - 'smoothness': Smoothness value
        """
        super().__init__(parent)
        self.images = images
        self.stats = stats
        self.setWindowTitle("Texture Feature Extraction Result")
        self.setMinimumSize(1000, 500)
        # Add minimize/maximize buttons
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI layout - 3x3 grid."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Table 2: Sample of Texture Feature Extraction Results")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Grid for images - 3x3 layout (2 rows)
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(10)
        
        # Image data: (key, label, row, col)
        image_data = [
            ('original', "Original Image", 0, 0),
            ('binarization', "Binarization", 0, 1),
            ('segmented', "Segmented Image", 0, 2),
            ('grayscale', "Grayscale Image", 1, 0),
            ('roi', "Region of Interest", 1, 1),
        ]
        
        for key, label_text, row, col in image_data:
            # Container for label + image
            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(5)
            container_layout.setContentsMargins(5, 5, 5, 5)
            
            # Label
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
            lbl.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(lbl)
            
            # Image - use ScalableImageLabel for resize support
            img = self.images.get(key)
            img_lbl = ScalableImageLabel(img)
            container_layout.addWidget(img_lbl, 1)  # stretch factor 1
            
            grid.addWidget(container, row, col)
        
        # Stats panel (row 1, col 2)
        stats_container = QWidget()
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setSpacing(5)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        
        stats_title = QLabel("Results of Mean,\nStandard Deviation\nand Smoothness")
        stats_title.setStyleSheet("font-weight: bold; font-size: 11px;")
        stats_title.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(stats_title)
        
        stats_text = f"""Mean: {self.stats.get('mean', 0):.4f}
Standard Deviation: {self.stats.get('std', 0):.4f}
Smoothness: {self.stats.get('smoothness', 0):.4f}"""
        
        stats_lbl = QLabel(stats_text)
        stats_lbl.setStyleSheet("background-color: #1a1a2e; border: 1px solid #444; padding: 15px; font-size: 12px;")
        stats_lbl.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(stats_lbl)
        
        grid.addWidget(stats_container, 1, 2)
        
        layout.addWidget(grid_widget)
        
        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("padding: 8px 20px;")
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: white; }
            QLabel { color: white; }
            QPushButton { 
                background-color: #4a4a5a; 
                color: white; 
                border: none; 
                border-radius: 5px; 
            }
            QPushButton:hover { background-color: #5a5a6a; }
        """)


def generate_segmentation_steps(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate intermediate segmentation images for Classical mode.
    
    Args:
        image: Input BGR image
        
    Returns:
        Dictionary with all intermediate images
    """
    results = {'original': image.copy()}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Sobel Edge Detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.uint8(255 * sobel / sobel.max()) if sobel.max() > 0 else np.uint8(sobel)
    results['sobel_edge'] = sobel
    
    # 2. Dilated Gradient Mask
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(sobel, kernel_small, iterations=2)
    results['dilated_gradient'] = dilated
    
    # 3. Filled Holes and Cleared Border
    # Use Otsu threshold on GRAYSCALE (not edges) for intensity-based segmentation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Use morphological closing to fill holes
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    # Use morphological opening to remove noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=2)
    # Clear border by setting edge pixels to 0
    border = 5
    cleared = opened.copy()
    cleared[:border, :] = 0
    cleared[-border:, :] = 0
    cleared[:, :border] = 0
    cleared[:, -border:] = 0
    results['filled_holes'] = cleared
    
    # 4. Erosion Gradient Mask - gentle erosion
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(cleared, kernel_erode, iterations=1)
    results['erosion_mask'] = eroded
    
    # 5. Segmented Image - remove small noise regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    min_area = max(100, eroded.size * 0.001)  # At least 0.1% of image or 100 pixels
    clean = np.zeros_like(eroded)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    results['segmented'] = clean
    
    return results


def generate_texture_steps(image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Generate intermediate texture feature extraction images for Classical mode.
    
    Args:
        image: Input BGR image
        
    Returns:
        Tuple of (images dict, stats dict)
    """
    images = {'original': image.copy()}
    
    # Convert to grayscale (for processing)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binarization using Otsu's method
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images['binarization'] = binary
    
    # Segmented image (apply binary mask to original)
    segmented = cv2.bitwise_and(image, image, mask=binary)
    images['segmented'] = segmented
    
    # Grayscale Image - grayscale version of the SEGMENTED image
    grayscale_segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    images['grayscale'] = grayscale_segmented
    
    # Region of Interest - zoomed crop of the grayscale segmented image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        # Add some padding for zoomed view
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
        
        # Crop the grayscale segmented image (zoomed to region)
        roi_crop = grayscale_segmented[y1:y2, x1:x2].copy()
        
        # Convert to 3 channel for display consistency
        roi = cv2.cvtColor(roi_crop, cv2.COLOR_GRAY2BGR)
    else:
        roi = cv2.cvtColor(grayscale_segmented, cv2.COLOR_GRAY2BGR)
    
    images['roi'] = roi
    
    # Calculate texture statistics on grayscale ROI
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        roi_gray = grayscale_segmented[y:y+h, x:x+w] if w > 0 and h > 0 else grayscale_segmented
        # Normalize to [0, 1]
        roi_norm = roi_gray.astype(float) / 255.0
        mean_val = np.mean(roi_norm)
        std_val = np.std(roi_norm)
        # Smoothness: 1 - 1/(1 + variance)
        variance = std_val ** 2
        smoothness = 1 - 1 / (1 + variance) if variance > 0 else 0
    else:
        mean_val = 0.0
        std_val = 0.0
        smoothness = 0.0
    
    stats = {
        'mean': mean_val,
        'std': std_val,
        'smoothness': smoothness
    }
    
    return images, stats

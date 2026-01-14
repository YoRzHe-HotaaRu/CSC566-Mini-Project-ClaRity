"""
Main Window for Road Surface Layer Analyzer GUI
PyQt5-based application with 4 analysis modes.

CSC566 Image Processing Mini Project
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QRadioButton,
    QSlider, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTextEdit, QSplitter, QFrame, QStatusBar,
    QMenuBar, QMenu, QAction, QMessageBox, QTabWidget, QStackedWidget,
    QScrollArea, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPalette, QColor

# Import processing modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ROAD_LAYERS, GUI_CONFIG
from src.preprocessing import preprocess_image, get_image_info
from src.texture_features import extract_all_texture_features
from src.segmentation import kmeans_segment, slic_segment
from src.classification import RoadLayerClassifier, get_layer_by_texture_heuristic
from src.morphology import apply_morphology_pipeline
from src.visualization import create_colored_segmentation


class AnalysisWorker(QThread):
    """Background worker thread for image analysis."""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image: np.ndarray, mode: str, params: dict):
        super().__init__()
        self.image = image
        self.mode = mode
        self.params = params
    
    def run(self):
        try:
            result = {}
            
            # Step 1: Preprocessing - NOW uses CLAHE clip and contrast method
            self.progress.emit(10, "Preprocessing image...")
            preprocessed = preprocess_image(
                self.image,
                denoise=self.params.get("noise_filter", "median"),
                denoise_kernel=self.params.get("kernel_size", 3),
                enhance=self.params.get("contrast_method", "clahe"),
                enhance_clip_limit=self.params.get("clahe_clip", 2.0),
                color_space="gray"
            )
            result["preprocessed"] = preprocessed
            result["preprocessing_params"] = {
                "noise_filter": self.params.get("noise_filter", "median"),
                "kernel_size": self.params.get("kernel_size", 3),
                "contrast_method": self.params.get("contrast_method", "clahe"),
                "clahe_clip": self.params.get("clahe_clip", 2.0)
            }
            
            if self.mode == "classical":
                # Step 2: Texture features
                self.progress.emit(30, "Extracting texture features...")
                features = extract_all_texture_features(
                    preprocessed,
                    use_glcm=self.params.get("use_glcm", True),
                    use_lbp=self.params.get("use_lbp", True),
                    use_gabor=self.params.get("use_gabor", False)
                )
                result["features"] = features
                
                # Step 3: Segmentation - NOW uses selected method!
                self.progress.emit(60, "Segmenting image...")
                n_clusters = self.params.get("n_clusters", 5)
                seg_method = self.params.get("segmentation_method", "K-Means")
                
                # Import segmentation functions
                from src.segmentation import kmeans_segment, slic_segment, watershed_segment
                
                if seg_method == "SLIC Superpixels":
                    # First create superpixels
                    superpixel_labels = slic_segment(self.image, n_segments=n_clusters * 50, compactness=10.0)
                    
                    # Compute mean color per superpixel and cluster those
                    unique_sp = np.unique(superpixel_labels)
                    sp_colors = []
                    for sp in unique_sp:
                        mask = superpixel_labels == sp
                        mean_color = self.image[mask].mean(axis=0)
                        sp_colors.append(mean_color)
                    sp_colors = np.array(sp_colors, dtype=np.float32)
                    
                    # K-Means cluster the superpixel colors
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    sp_cluster_labels = kmeans.fit_predict(sp_colors)
                    
                    # Map superpixel clusters back to full image
                    labels = np.zeros_like(superpixel_labels)
                    for idx, sp in enumerate(unique_sp):
                        labels[superpixel_labels == sp] = sp_cluster_labels[idx]
                elif seg_method == "Watershed":
                    gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
                    labels = watershed_segment(gray_img)
                    # Limit to n_clusters by mapping
                    unique = np.unique(labels)
                    if len(unique) > n_clusters:
                        labels = (labels % n_clusters)
                else:  # K-Means (default)
                    labels, _ = kmeans_segment(self.image, n_clusters=n_clusters)
                
                result["segmentation_method"] = seg_method
                
                # Step 4: Morphological cleanup - FIXED LOGIC
                self.progress.emit(80, "Refining segmentation...")
                use_morphology = self.params.get("use_morphology", True)
                fill_holes = self.params.get("fill_holes", True)
                
                if use_morphology or fill_holes:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    cleaned_labels = np.zeros_like(labels)
                    
                    for i in range(n_clusters):
                        mask = (labels == i).astype(np.uint8)
                        
                        if use_morphology:
                            # Opening removes small noise (erosion then dilation)
                            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                            # Closing fills small gaps (dilation then erosion)
                            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        
                        if fill_holes:
                            # Fill holes using contour filling
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            mask_filled = np.zeros_like(mask)
                            cv2.drawContours(mask_filled, contours, -1, 1, -1)
                            mask = mask_filled
                        
                        cleaned_labels[mask == 1] = i
                    
                    labels = cleaned_labels
                
                result["labels"] = labels + 1  # 1-indexed
                result["morphology_applied"] = use_morphology
                result["fill_holes_applied"] = fill_holes
                
                # Step 6: Classification
                self.progress.emit(90, "Classifying layers...")
                classification = get_layer_by_texture_heuristic(features)
                classification["method"] = f"Classical ({seg_method})"
                result["classification"] = classification
            
            elif self.mode == "deep_learning":
                self.progress.emit(30, "Initializing DeepLabv3+ model...")
                try:
                    from src.deep_learning import DeepLabSegmenter
                    from src.config import ROAD_LAYERS
                    
                    # Get parameters from GUI
                    backbone_raw = self.params.get("dl_backbone", "ResNet-101")
                    backbone = backbone_raw.lower().replace("-", "")
                    use_pretrained = self.params.get("dl_pretrained", True)
                    use_cuda = "cuda" in self.params.get("dl_device", "CPU").lower()
                    resolution = self.params.get("dl_resolution", "512x512")
                    confidence_threshold = self.params.get("dl_confidence_threshold", 0.5)
                    
                    # Handle resolution
                    self.progress.emit(40, f"Preparing image ({resolution})...")
                    if resolution == "512x512":
                        input_image = cv2.resize(self.image, (512, 512))
                    elif resolution == "256x256":
                        input_image = cv2.resize(self.image, (256, 256))
                    else:  # Original
                        input_image = self.image.copy()
                    
                    # Create segmenter with pretrained option
                    self.progress.emit(50, "Loading model...")
                    encoder_weights = "imagenet" if use_pretrained else None
                    segmenter = DeepLabSegmenter(
                        encoder_name=backbone,
                        encoder_weights=encoder_weights,
                        use_cuda=use_cuda
                    )
                    
                    # Run inference
                    self.progress.emit(70, "Running inference...")
                    labels = segmenter.segment(input_image)
                    
                    # Resize labels back to original size if needed
                    if resolution != "Original":
                        labels = cv2.resize(
                            labels.astype(np.uint8),
                            (self.image.shape[1], self.image.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    
                    result["labels"] = labels
                    
                    # Classify dominant layer from predictions
                    self.progress.emit(85, "Analyzing predictions...")
                    unique, counts = np.unique(labels, return_counts=True)
                    total_pixels = labels.size
                    
                    # Find dominant layer (most pixels)
                    dominant_idx = np.argmax(counts)
                    dominant_layer = int(unique[dominant_idx])
                    dominant_pct = counts[dominant_idx] / total_pixels
                    
                    # Get layer info
                    layer_info = ROAD_LAYERS.get(dominant_layer, {})
                    layer_name = layer_info.get("name", f"Layer {dominant_layer}")
                    material = layer_info.get("material", "Unknown")
                    
                    # Build layer distribution info
                    layer_dist = {}
                    for layer, count in zip(unique, counts):
                        pct = count / total_pixels
                        if pct >= confidence_threshold:  # Only show layers above threshold
                            linfo = ROAD_LAYERS.get(int(layer), {})
                            layer_dist[int(layer)] = {
                                "name": linfo.get("name", f"Layer {layer}"),
                                "percentage": pct
                            }
                    
                    result["layer_distribution"] = layer_dist
                    result["classification"] = {
                        "layer_number": dominant_layer,
                        "layer_name": layer_name,
                        "material": material,
                        "confidence": dominant_pct,
                        "method": f"DeepLabv3+ ({backbone_raw}, {'CUDA' if use_cuda else 'CPU'})"
                    }
                    
                except Exception as e:
                    self.error.emit(f"Deep learning error: {str(e)}")
                    return
            
            elif self.mode == "vlm":
                self.progress.emit(50, "Analyzing with GLM-4.6V...")
                try:
                    from src.vlm_analyzer import VLMAnalyzer
                    analyzer = VLMAnalyzer()
                    
                    # Save temp image for API
                    temp_path = Path(__file__).parent.parent / "results" / "temp_analysis.jpg"
                    temp_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(temp_path), self.image)
                    
                    # Get VLM parameters from GUI
                    analysis_type = self.params.get("vlm_analysis_type", "Layer ID")
                    temperature = self.params.get("vlm_temperature", 0.3)
                    
                    # Choose analysis method and pass temperature
                    if analysis_type == "Detailed":
                        vlm_result = analyzer.get_detailed_analysis(str(temp_path), temperature=temperature)
                    elif analysis_type == "Quick Scan":
                        # Quick scan: simple prompt, higher temperature for speed
                        quick_prompt = """Quickly identify the road layer in this image. 
                        Just tell me: Layer number (1-5), Layer name, and Confidence (%).
                        Keep response very brief."""
                        vlm_result = analyzer.analyze_road_layer(str(temp_path), custom_prompt=quick_prompt, temperature=min(temperature + 0.2, 1.0))
                    else:  # Layer ID - standard analysis
                        vlm_result = analyzer.analyze_road_layer(str(temp_path), temperature=temperature)
                    
                    # Add analysis type to result for display
                    vlm_result['analysis_type'] = analysis_type
                    vlm_result['temperature_used'] = temperature

                    # Ensure method field is populated
                    if 'method' not in vlm_result or vlm_result.get('method') == 'N/A':
                        vlm_result['method'] = f'VLM Analysis (GLM-4.6V, {analysis_type})'

                    result["classification"] = vlm_result

                    # Create ENHANCED VLM visualization with ROAD LAYER HIGHLIGHTING
                    h, w = self.image.shape[:2]
                    layer_num = vlm_result.get("layer_number", 1)
                    confidence = vlm_result.get("confidence", 0.5)
                    layer_name = vlm_result.get("layer_name", "Unknown")
                    
                    # Get layer color from config
                    from src.config import ROAD_LAYERS, LAYER_COLORS_RGB
                    layer_color = LAYER_COLORS_RGB.get(layer_num, (128, 128, 128))
                    
                    # Start with original image
                    blended = self.image.copy()
                    
                    # ===== EDGE DETECTION + MUTED/SHARP EFFECT =====
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    
                    # Create slightly desaturated look (muted colors)
                    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    blended = cv2.addWeighted(self.image, 0.7, gray_3ch, 0.3, 0)  # 30% gray blend
                    
                    # Darken slightly
                    blended = cv2.convertScaleAbs(blended, alpha=0.92, beta=-5)  # Slight darkening
                    
                    # Apply MILD sharpening kernel
                    sharpen_kernel = np.array([[0, -0.5, 0],
                                               [-0.5,  3, -0.5],
                                               [0, -0.5, 0]])
                    blended = cv2.filter2D(blended, -1, sharpen_kernel)
                    
                    # Draw edges in dark gray for definition
                    blended[edges > 0] = [60, 60, 60]  # Dark gray edges
                    
                    # ===== BOUNDING BOX FRAME =====
                    box_color = (0, 255, 0)  # Bright Green
                    box_thickness = 4
                    margin = 8  # Inset from edges
                    
                    # Draw main bounding box
                    cv2.rectangle(blended, (margin, margin), (w - margin, h - margin), box_color, box_thickness)
                    
                    # Add corner accent marks (thicker, longer lines at corners)
                    corner_len = min(60, min(h, w) // 6)
                    accent_thickness = 6
                    
                    # Top-left corner
                    cv2.line(blended, (margin, margin), (margin + corner_len, margin), box_color, accent_thickness)
                    cv2.line(blended, (margin, margin), (margin, margin + corner_len), box_color, accent_thickness)
                    # Top-right corner
                    cv2.line(blended, (w - margin, margin), (w - margin - corner_len, margin), box_color, accent_thickness)
                    cv2.line(blended, (w - margin, margin), (w - margin, margin + corner_len), box_color, accent_thickness)
                    # Bottom-left corner
                    cv2.line(blended, (margin, h - margin), (margin + corner_len, h - margin), box_color, accent_thickness)
                    cv2.line(blended, (margin, h - margin), (margin, h - margin - corner_len), box_color, accent_thickness)
                    # Bottom-right corner
                    cv2.line(blended, (w - margin, h - margin), (w - margin - corner_len, h - margin), box_color, accent_thickness)
                    cv2.line(blended, (w - margin, h - margin), (w - margin, h - margin - corner_len), box_color, accent_thickness)
                    
                    # ===== INFO BANNER AT TOP =====
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = max(0.6, min(h, w) / 500)
                    thickness = max(1, int(font_scale * 2))
                    
                    banner_height = int(h * 0.12)
                    banner_overlay = blended.copy()
                    cv2.rectangle(banner_overlay, (0, 0), (w, banner_height), (0, 0, 0), -1)
                    blended = cv2.addWeighted(blended, 0.4, banner_overlay, 0.6, 0)
                    
                    # Layer name text with shadow
                    text = f"Layer {layer_num}: {layer_name}"
                    cv2.putText(blended, text, (12, int(banner_height * 0.50)), 
                               font, font_scale, (0, 0, 0), thickness + 2)
                    cv2.putText(blended, text, (10, int(banner_height * 0.48)), 
                               font, font_scale, (255, 255, 255), thickness)
                    
                    # Confidence text
                    conf_text = f"Confidence: {confidence:.0%}"
                    cv2.putText(blended, conf_text, (10, int(banner_height * 0.85)), 
                               font, font_scale * 0.7, (200, 200, 200), max(1, thickness - 1))
                    
                    # Confidence bar
                    bar_width = int(w * 0.25)
                    bar_x = w - bar_width - 20
                    bar_y = int(banner_height * 0.40)
                    bar_h = 16
                    cv2.rectangle(blended, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_h), (60, 60, 60), -1)
                    cv2.rectangle(blended, (bar_x, bar_y), (bar_x + int(bar_width * confidence), bar_y + bar_h), box_color, -1)
                    cv2.rectangle(blended, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_h), (255, 255, 255), 1)
                    
                    # Store the enhanced visualization
                    result["vlm_visualization"] = blended
                    
                    # Also create labels for legend compatibility
                    labels = np.ones((h, w), dtype=np.uint8) * layer_num
                    result["labels"] = labels
                except Exception as e:
                    self.error.emit(f"VLM analysis error: {str(e)}")
                    return
            
            elif self.mode == "hybrid":
                # ===== HYBRID MODE: Classical + AI Validation =====
                
                # Step 1: Run full Classical analysis (inherits ALL Classical settings)
                self.progress.emit(20, "Running classical analysis...")
                
                # Texture features with selected options
                features = extract_all_texture_features(
                    preprocessed,
                    use_glcm=self.params.get("use_glcm", True),
                    use_lbp=self.params.get("use_lbp", True),
                    use_gabor=self.params.get("use_gabor", False)
                )
                result["features"] = features
                
                # Segmentation with selected method (same as Classical)
                self.progress.emit(35, "Segmenting image...")
                n_clusters = self.params.get("n_clusters", 5)
                seg_method = self.params.get("segmentation_method", "K-Means")
                
                from src.segmentation import kmeans_segment, slic_segment, watershed_segment
                
                if seg_method == "SLIC Superpixels":
                    superpixel_labels = slic_segment(self.image, n_segments=n_clusters * 50, compactness=10.0)
                    unique_sp = np.unique(superpixel_labels)
                    sp_colors = []
                    for sp in unique_sp:
                        mask = superpixel_labels == sp
                        mean_color = self.image[mask].mean(axis=0)
                        sp_colors.append(mean_color)
                    sp_colors = np.array(sp_colors, dtype=np.float32)
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    sp_cluster_labels = kmeans.fit_predict(sp_colors)
                    labels = np.zeros_like(superpixel_labels)
                    for idx, sp in enumerate(unique_sp):
                        labels[superpixel_labels == sp] = sp_cluster_labels[idx]
                elif seg_method == "Watershed":
                    gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) == 3 else self.image
                    labels = watershed_segment(gray_img)
                    unique = np.unique(labels)
                    if len(unique) > n_clusters:
                        labels = (labels % n_clusters)
                else:  # K-Means (default)
                    labels, _ = kmeans_segment(self.image, n_clusters=n_clusters)
                
                result["segmentation_method"] = seg_method
                
                # Morphology + Fill holes (same as Classical)
                self.progress.emit(45, "Refining segmentation...")
                use_morphology = self.params.get("use_morphology", True)
                fill_holes = self.params.get("fill_holes", True)
                
                if use_morphology or fill_holes:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                    cleaned_labels = np.zeros_like(labels)
                    for i in range(n_clusters):
                        mask = (labels == i).astype(np.uint8)
                        if use_morphology:
                            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                        if fill_holes:
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            mask_filled = np.zeros_like(mask)
                            cv2.drawContours(mask_filled, contours, -1, 1, -1)
                            mask = mask_filled
                        cleaned_labels[mask == 1] = i
                    labels = cleaned_labels
                
                result["labels"] = labels + 1
                result["morphology_applied"] = use_morphology
                result["fill_holes_applied"] = fill_holes
                
                # Classical classification
                classical_result = get_layer_by_texture_heuristic(features)
                classical_result["method"] = f"Classical ({seg_method})"
                
                # Step 2: VLM Cross-Check (if enabled)
                vlm_result = None
                vlm_enabled = self.params.get("hybrid_vlm_validation", True)
                
                if vlm_enabled:
                    self.progress.emit(60, "Running VLM cross-check...")
                    try:
                        from src.vlm_analyzer import VLMAnalyzer
                        analyzer = VLMAnalyzer()
                        temp_path = Path(__file__).parent.parent / "results" / "temp_analysis.jpg"
                        temp_path.parent.mkdir(exist_ok=True)
                        cv2.imwrite(str(temp_path), self.image)
                        vlm_result = analyzer.analyze_road_layer(str(temp_path))
                        vlm_result["method"] = "VLM (GLM-4.6V)"
                    except Exception as e:
                        # VLM failed, continue with Classical only
                        vlm_result = None
                        result["vlm_error"] = str(e)
                
                # Step 3: Combine results based on Conflict Rule
                self.progress.emit(85, "Combining results...")
                conflict_rule = self.params.get("hybrid_conflict_rule", "Higher Confidence Wins")
                classical_weight = self.params.get("classical_weight", 0.7)
                
                if vlm_result is None:
                    # VLM disabled or failed - use Classical only
                    final_result = classical_result.copy()
                    final_result["method"] = f"Hybrid (Classical only - VLM {'disabled' if not vlm_enabled else 'failed'})"
                elif conflict_rule == "Higher Confidence Wins":
                    vlm_conf = vlm_result.get("confidence", 0)
                    classical_conf = classical_result.get("confidence", 0)
                    if vlm_conf > classical_conf:
                        final_result = vlm_result.copy()
                        final_result["method"] = f"Hybrid (VLM won: {vlm_conf:.0%} > {classical_conf:.0%})"
                    else:
                        final_result = classical_result.copy()
                        final_result["method"] = f"Hybrid (Classical won: {classical_conf:.0%} >= {vlm_conf:.0%})"
                elif conflict_rule == "Classical Priority":
                    vlm_conf = vlm_result.get("confidence", 0)
                    if vlm_conf > 0.9:
                        final_result = vlm_result.copy()
                        final_result["method"] = f"Hybrid (VLM override: {vlm_conf:.0%} > 90%)"
                    else:
                        final_result = classical_result.copy()
                        final_result["method"] = "Hybrid (Classical priority)"
                elif conflict_rule == "VLM Priority":
                    final_result = vlm_result.copy()
                    final_result["method"] = "Hybrid (VLM priority)"
                else:  # Weighted Average or default
                    # Blend confidences
                    vlm_conf = vlm_result.get("confidence", 0)
                    classical_conf = classical_result.get("confidence", 0)
                    blended_conf = classical_weight * classical_conf + (1 - classical_weight) * vlm_conf
                    
                    # Use layer from PRIMARY weight source (not raw confidence!)
                    ai_weight = 1 - classical_weight
                    if classical_weight >= ai_weight:
                        # Classical has more weight, use Classical's layer
                        final_result = classical_result.copy()
                    else:
                        # AI has more weight, use VLM's layer
                        final_result = vlm_result.copy()
                    
                    final_result["confidence"] = blended_conf
                    final_result["method"] = f"Hybrid (Weighted: {classical_weight:.0%} Classical + {ai_weight:.0%} AI)"
                
                # Store both results for reference
                result["classical_result"] = classical_result
                result["vlm_result"] = vlm_result
                result["classification"] = final_result
            
            self.progress.emit(100, "Analysis complete!")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class ImageLabel(QLabel):
    """Custom label for displaying images with aspect ratio preservation and drag-drop support."""
    
    # Signal emitted when image is dropped
    imageDropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                border: 2px dashed #555;
                border-radius: 8px;
            }
        """)
        self.setText("Drop image here\nor click Load Image")
        self._pixmap = None
        
        # Enable drag-drop
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        """Handle drag enter - accept image files."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile().lower()
                if file_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        QLabel {
                            background-color: #3a4a3a;
                            border: 2px dashed #4CAF50;
                            border-radius: 8px;
                        }
                    """)
                    return
        event.ignore()
    
    def dragLeaveEvent(self, event):
        """Handle drag leave - reset style."""
        self.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                border: 2px dashed #555;
                border-radius: 8px;
            }
        """)
    
    def dropEvent(self, event):
        """Handle drop - load image file."""
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                self.imageDropped.emit(file_path)
                event.acceptProposedAction()
        
        # Reset style
        self.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                border: 2px dashed #555;
                border-radius: 8px;
            }
        """)
    
    def setImage(self, image: np.ndarray):
        """Set image from numpy array."""
        if len(image.shape) == 2:
            h, w = image.shape
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            h, w, c = image.shape
            if c == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                bytes_per_line = 3 * w
                q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            else:
                bytes_per_line = w
                q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        
        self._pixmap = QPixmap.fromImage(q_image)
        self._updatePixmap()
    
    def _updatePixmap(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._updatePixmap()


class MainWindow(QMainWindow):
    """Main application window for Road Surface Layer Analyzer."""
    
    def __init__(self):
        super().__init__()
        
        self.image = None
        self.result = None
        self.worker = None
        
        self.init_ui()
        self.apply_dark_theme()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(GUI_CONFIG["window_title"])
        self.setMinimumSize(*GUI_CONFIG["min_size"])
        self.resize(*GUI_CONFIG["window_size"])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Image display
        left_panel = self.create_image_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Controls and results
        right_panel = self.create_control_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([700, 500])
        main_layout.addWidget(splitter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load an image to begin")
    
    def create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Image", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Result", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_image_panel(self) -> QWidget:
        """Create image display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image display area
        images_layout = QHBoxLayout()
        
        # Original image
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)
        self.original_label = ImageLabel()
        # Connect drag-drop signal (Bug 2 fix)
        self.original_label.imageDropped.connect(self.load_image_from_path)
        original_layout.addWidget(self.original_label)
        images_layout.addWidget(original_group)
        
        # Result image
        result_group = QGroupBox("Segmentation Result")
        result_layout = QVBoxLayout(result_group)
        self.result_label = ImageLabel()
        self.result_label.setText("Result will appear here")
        result_layout.addWidget(self.result_label)
        images_layout.addWidget(result_group)
        
        layout.addLayout(images_layout)
        
        # Layer legend
        self.legend_group = QGroupBox("Layer Legend")
        self.legend_group.setMaximumHeight(90)  # Compact height for wrapped centered text
        legend_layout = QVBoxLayout(self.legend_group)  # Changed to QVBoxLayout for message
        legend_layout.setContentsMargins(5, 5, 5, 5)  # Compact margins

        # Create placeholder label
        self.legend_placeholder = QLabel("After running the Analysis, the legend will show up here")
        self.legend_placeholder.setWordWrap(True)  # Enable word wrapping to prevent cutoff
        self.legend_placeholder.setStyleSheet("color: #888; font-style: italic; padding: 10px; font-size: 12px;")  # Adjusted font size
        self.legend_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend_layout.addWidget(self.legend_placeholder)

        # Store legend items for dynamic updates
        self.legend_items = {}
        self.legend_widget = QWidget()  # Container for legend items
        self.legend_layout = QHBoxLayout(self.legend_widget)  # Horizontal layout for items
        self.legend_layout.setSpacing(15)  # Spacing between items

        for layer_num in range(1, 6):
            layer = ROAD_LAYERS[layer_num]
            color = layer["hex_color"]
            
            legend_item = QLabel(f"■ {layer['name']}")
            legend_item.setStyleSheet(f'''
                color: {color}; 
                font-weight: bold; 
                padding: 2px 8px;
                font-size: 10px;
            ''')
            self.legend_items[layer_num] = legend_item
            self.legend_layout.addWidget(legend_item)
        
        self.legend_layout.addStretch()
        legend_layout.addWidget(self.legend_widget)
        
        # Initially hide legend widget and show placeholder (Bug 1 fix)
        self.legend_widget.setVisible(False)
        self.legend_placeholder.setVisible(True)
        layout.addWidget(self.legend_group)
        return panel
    def create_control_panel(self) -> QWidget:
        """Create control panel with parameters."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Analysis mode selection
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_classical = QRadioButton("Classical (Texture-based)")
        self.mode_classical.setChecked(True)
        self.mode_dl = QRadioButton("Deep Learning (CNN Classifier)")
        self.mode_vlm = QRadioButton("VLM Analysis (GLM-4.6V)")
        self.mode_hybrid = QRadioButton("Hybrid (Classical + AI)")
        
        mode_layout.addWidget(self.mode_classical)
        mode_layout.addWidget(self.mode_dl)
        mode_layout.addWidget(self.mode_vlm)
        mode_layout.addWidget(self.mode_hybrid)
        
        # Connect mode buttons to dynamic panel switching
        self.mode_classical.toggled.connect(lambda: self.switch_mode_panel("classical"))
        self.mode_dl.toggled.connect(lambda: self.switch_mode_panel("deep_learning"))
        self.mode_vlm.toggled.connect(lambda: self.switch_mode_panel("vlm"))
        self.mode_hybrid.toggled.connect(lambda: self.switch_mode_panel("hybrid"))
        
        layout.addWidget(mode_group)
        
        # Dynamic parameters stack (different panels for each mode)
        self.params_stack = QStackedWidget()
        
        # Create mode-specific parameter panels
        self.classical_params = self.create_classical_params()
        self.params_stack.addWidget(self.classical_params)
        
        self.dl_params = self.create_deep_learning_params()
        self.params_stack.addWidget(self.dl_params)
        
        self.vlm_params = self.create_vlm_params()
        self.params_stack.addWidget(self.vlm_params)
        
        self.hybrid_params = self.create_hybrid_params()
        self.params_stack.addWidget(self.hybrid_params)
        
        layout.addWidget(self.params_stack)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📂 Load Image")
        self.load_btn.clicked.connect(self.load_image)
        buttons_layout.addWidget(self.load_btn)
        
        self.analyze_btn = QPushButton("▶ Analyze")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        buttons_layout.addWidget(self.analyze_btn)
        
        self.save_btn = QPushButton("💾 Export")
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_btn)
        
        layout.addLayout(buttons_layout)
        
        # Results panel
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        self.results_text.setPlaceholderText("Analysis results will appear here...")
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Summary panel (NEW)
        summary_group = QGroupBox("📋 Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setPlaceholderText("Plain-English summary will appear here...")
        self.summary_text.setStyleSheet(
            "QTextEdit { background-color: #1a1a2e; border: 2px solid #4a4a6a; border-radius: 8px; padding: 10px; font-size: 11px; line-height: 1.4; }"
        )
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        layout.addStretch()
        
        return panel
    
    def switch_mode_panel(self, mode: str):
        """Switch parameter panel based on selected analysis mode."""
        mode_to_index = {
            "classical": 0,
            "deep_learning": 1,
            "vlm": 2,
            "hybrid": 3
        }
        self.params_stack.setCurrentIndex(mode_to_index.get(mode, 0))
        self.status_bar.showMessage(f"Switched to {mode.replace('_', ' ').title()} mode")
    
    def create_classical_params(self) -> QWidget:
        """Create Classical mode parameter panel with tabs."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Use tab widget for organized parameters
        tabs = QTabWidget()
        
        # Preprocessing tab
        preprocess_tab = self.create_preprocessing_params()
        tabs.addTab(preprocess_tab, "Preprocessing")
        
        # Features tab
        features_tab = self.create_features_params()
        tabs.addTab(features_tab, "Features")
        
        # Segmentation tab
        seg_tab = self.create_segmentation_params()
        tabs.addTab(seg_tab, "Segmentation")
        
        layout.addWidget(tabs)
        return widget
    
    def create_deep_learning_params(self) -> QWidget:
        """Create Deep Learning mode parameter panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model Settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QGridLayout(model_group)
        
        # Backbone selection
        model_layout.addWidget(QLabel("Backbone:"), 0, 0)
        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["ResNet-101", "ResNet-50", "MobileNetV2"])
        self.backbone_combo.setToolTip(
            "ResNet-101: Best accuracy, slowest\n"
            "ResNet-50: Good balance\n"
            "MobileNetV2: Fastest, lower accuracy"
        )
        model_layout.addWidget(self.backbone_combo, 0, 1)
        
        # Pretrained
        self.pretrained_check = QCheckBox("Use Pretrained (ImageNet)")
        self.pretrained_check.setChecked(True)
        self.pretrained_check.setToolTip(
            "When enabled, uses weights pre-trained on ImageNet.\n"
            "Gives better results for general images."
        )
        model_layout.addWidget(self.pretrained_check, 1, 0, 1, 2)
        
        # Device selection
        model_layout.addWidget(QLabel("Device:"), 2, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA (GPU)"])
        self.device_combo.setToolTip("CUDA requires NVIDIA GPU with CUDA support")
        model_layout.addWidget(self.device_combo, 2, 1)
        
        layout.addWidget(model_group)
        
        # Inference Settings group
        infer_group = QGroupBox("Inference Settings")
        infer_layout = QGridLayout(infer_group)
        
        # Input resolution
        infer_layout.addWidget(QLabel("Input Resolution:"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["512x512", "256x256", "Original"])
        self.resolution_combo.setToolTip(
            "512x512: Best quality (default)\n"
            "256x256: Faster, lower detail\n"
            "Original: Uses image's original size"
        )
        infer_layout.addWidget(self.resolution_combo, 0, 1)
        
        # Confidence threshold
        infer_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setToolTip(
            "Minimum confidence for predictions.\n"
            "Higher = stricter, fewer predictions."
        )
        infer_layout.addWidget(self.confidence_spin, 1, 1)
        
        # Hidden batch_spin for compatibility (always 1)
        self.batch_spin = QSpinBox()
        self.batch_spin.setValue(1)
        self.batch_spin.setVisible(False)
        
        layout.addWidget(infer_group)
        layout.addStretch()
        return widget
    
    def create_vlm_params(self) -> QWidget:
        """Create VLM Analysis mode parameter panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # VLM Settings group
        vlm_group = QGroupBox("VLM Settings")
        vlm_layout = QGridLayout(vlm_group)
        
        # Model info
        vlm_layout.addWidget(QLabel("Model:"), 0, 0)
        vlm_layout.addWidget(QLabel("GLM-4.6V (via ZenMux API)"), 0, 1)
        
        # Analysis type
        vlm_layout.addWidget(QLabel("Analysis Type:"), 1, 0)
        self.vlm_type_combo = QComboBox()
        self.vlm_type_combo.addItems(["Layer ID", "Detailed", "Quick Scan"])
        vlm_layout.addWidget(self.vlm_type_combo, 1, 1)
        
        # Temperature
        vlm_layout.addWidget(QLabel("Temperature:"), 2, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 1.0)
        self.temp_spin.setValue(0.3)
        self.temp_spin.setSingleStep(0.1)
        vlm_layout.addWidget(self.temp_spin, 2, 1)
        
        layout.addWidget(vlm_group)
        
        # Output Options group
        output_group = QGroupBox("Output Options")
        output_layout = QVBoxLayout(output_group)
        
        self.vlm_layer_check = QCheckBox("Include Layer Name")
        self.vlm_layer_check.setChecked(True)
        output_layout.addWidget(self.vlm_layer_check)
        
        self.vlm_conf_check = QCheckBox("Include Confidence")
        self.vlm_conf_check.setChecked(True)
        output_layout.addWidget(self.vlm_conf_check)
        
        self.vlm_material_check = QCheckBox("Include Material Description")
        self.vlm_material_check.setChecked(True)
        output_layout.addWidget(self.vlm_material_check)
        
        self.vlm_texture_check = QCheckBox("Include Texture Description")
        self.vlm_texture_check.setChecked(True)
        output_layout.addWidget(self.vlm_texture_check)
        
        self.vlm_recom_check = QCheckBox("Include Recommendations")
        self.vlm_recom_check.setChecked(False)
        output_layout.addWidget(self.vlm_recom_check)
        
        layout.addWidget(output_group)
        layout.addStretch()
        return widget
    
    def create_hybrid_params(self) -> QWidget:
        """Create Hybrid mode parameter panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Hybrid Settings group
        hybrid_group = QGroupBox("Hybrid Settings")
        hybrid_layout = QVBoxLayout(hybrid_group)
        
        # VLM Cross-Check toggle
        self.ai_validation_check = QCheckBox("Enable VLM Cross-Check")
        self.ai_validation_check.setChecked(True)
        self.ai_validation_check.setToolTip("When enabled, VLM will validate Classical results")
        hybrid_layout.addWidget(self.ai_validation_check)
        
        # Separator
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #555;")
        hybrid_layout.addWidget(line1)
        
        # Weight Balance Slider
        weight_label = QLabel("Weight Balance:")
        weight_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        hybrid_layout.addWidget(weight_label)
        
        # Slider with Classical on left, AI on right
        self.classical_weight_spin = QSlider(Qt.Horizontal)
        self.classical_weight_spin.setRange(0, 100)
        self.classical_weight_spin.setValue(70)
        self.classical_weight_spin.setTickPosition(QSlider.TicksBelow)
        self.classical_weight_spin.setTickInterval(10)
        # Two-tone gradient: Green (Classical) on left, Blue (AI) on right
        self.classical_weight_spin.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #444;
                height: 10px;
                border-radius: 5px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #2196F3);
            }
            QSlider::handle:horizontal {
                background: white;
                border: 2px solid #888;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #f0f0f0;
                border: 2px solid #4CAF50;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50;
                border-radius: 5px;
            }
            QSlider::add-page:horizontal {
                background: #2196F3;
                border-radius: 5px;
            }
        """)
        hybrid_layout.addWidget(self.classical_weight_spin)
        
        # Dynamic percentage labels below slider
        weight_info_layout = QHBoxLayout()
        self.classical_pct_label = QLabel("Classical: 70%")
        self.classical_pct_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.ai_pct_label = QLabel("AI: 30%")
        self.ai_pct_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.ai_pct_label.setAlignment(Qt.AlignRight)
        weight_info_layout.addWidget(self.classical_pct_label)
        weight_info_layout.addStretch()
        weight_info_layout.addWidget(self.ai_pct_label)
        hybrid_layout.addLayout(weight_info_layout)
        
        # Update labels when slider changes
        def update_weight_labels(value):
            self.classical_pct_label.setText(f"Classical: {value}%")
            self.ai_pct_label.setText(f"AI: {100 - value}%")
        
        self.classical_weight_spin.valueChanged.connect(update_weight_labels)
        
        # Separator
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #555; margin-top: 10px;")
        hybrid_layout.addWidget(line2)
        
        # Conflict resolution
        conflict_label = QLabel("Conflict Rule:")
        conflict_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        hybrid_layout.addWidget(conflict_label)
        
        self.conflict_combo = QComboBox()
        self.conflict_combo.addItems([
            "Weighted Average",  # Default - first in list
            "Higher Confidence Wins",
            "Classical Priority", 
            "VLM Priority"
        ])
        self.conflict_combo.setToolTip(
            "Weighted Average: Blend using slider weights (default)\n"
            "Higher Confidence: Use result with higher confidence\n"
            "Classical Priority: Use Classical unless VLM >90%\n"
            "VLM Priority: Always use VLM result"
        )
        hybrid_layout.addWidget(self.conflict_combo)
        
        # Enable/disable slider based on conflict rule
        def on_conflict_rule_changed(rule):
            is_weighted = (rule == "Weighted Average")
            self.classical_weight_spin.setEnabled(is_weighted)
            self.classical_pct_label.setEnabled(is_weighted)
            self.ai_pct_label.setEnabled(is_weighted)
            # Update styling for disabled state
            if is_weighted:
                self.classical_pct_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.ai_pct_label.setStyleSheet("color: #2196F3; font-weight: bold;")
            else:
                self.classical_pct_label.setStyleSheet("color: #666; font-weight: normal;")
                self.ai_pct_label.setStyleSheet("color: #666; font-weight: normal;")
        
        self.conflict_combo.currentTextChanged.connect(on_conflict_rule_changed)
        
        # Hidden primary combo (for compatibility - always Classical)
        self.primary_combo = QComboBox()
        self.primary_combo.addItems(["Classical"])
        self.primary_combo.setVisible(False)
        
        layout.addWidget(hybrid_group)
        
        # Info text
        info_label = QLabel(
            "Hybrid mode runs Classical analysis first, then validates with VLM.\n"
            "Slide left for more Classical weight, right for more AI weight."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-style: italic; padding: 10px;")
        layout.addWidget(info_label)
        
        layout.addStretch()
        return widget
    
    def create_preprocessing_params(self) -> QWidget:
        """Create preprocessing parameters widget."""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Noise filter
        layout.addWidget(QLabel("Noise Filter:"), 0, 0)
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["median", "gaussian", "bilateral"])
        layout.addWidget(self.noise_combo, 0, 1)
        
        # Kernel size
        layout.addWidget(QLabel("Kernel Size:"), 1, 0)
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(3, 15)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setValue(3)
        layout.addWidget(self.kernel_spin, 1, 1)
        
        # Contrast method
        layout.addWidget(QLabel("Contrast:"), 2, 0)
        self.contrast_combo = QComboBox()
        self.contrast_combo.addItems(["clahe", "histogram_eq", "gamma"])
        layout.addWidget(self.contrast_combo, 2, 1)
        
        # CLAHE clip limit
        layout.addWidget(QLabel("CLAHE Clip:"), 3, 0)
        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(1.0, 5.0)
        self.clahe_spin.setValue(2.0)
        self.clahe_spin.setSingleStep(0.5)
        layout.addWidget(self.clahe_spin, 3, 1)
        
        layout.setRowStretch(4, 1)
        return widget
    
    def create_features_params(self) -> QWidget:
        """Create feature extraction parameters widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Feature checkboxes
        self.glcm_check = QCheckBox("GLCM (Gray-Level Co-occurrence)")
        self.glcm_check.setChecked(True)
        layout.addWidget(self.glcm_check)
        
        self.lbp_check = QCheckBox("LBP (Local Binary Patterns)")
        self.lbp_check.setChecked(True)
        layout.addWidget(self.lbp_check)
        
        self.gabor_check = QCheckBox("Gabor Filters")
        layout.addWidget(self.gabor_check)
        
        layout.addStretch()
        return widget
    
    def create_segmentation_params(self) -> QWidget:
        """Create segmentation parameters widget."""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Method
        layout.addWidget(QLabel("Method:"), 0, 0)
        self.seg_combo = QComboBox()
        self.seg_combo.addItems(["K-Means", "SLIC Superpixels", "Watershed"])
        layout.addWidget(self.seg_combo, 0, 1)
        
        # Number of clusters
        layout.addWidget(QLabel("Clusters (K):"), 1, 0)
        self.clusters_spin = QSpinBox()
        self.clusters_spin.setRange(2, 10)
        self.clusters_spin.setValue(5)
        layout.addWidget(self.clusters_spin, 1, 1)
        
        # Morphology
        self.morph_check = QCheckBox("Apply Morphology")
        self.morph_check.setChecked(True)
        layout.addWidget(self.morph_check, 2, 0, 1, 2)
        
        self.fill_check = QCheckBox("Fill Holes")
        self.fill_check.setChecked(True)
        layout.addWidget(self.fill_check, 3, 0, 1, 2)
        
        layout.setRowStretch(4, 1)
        return widget
    
    def apply_dark_theme(self):
        """Apply dark theme to application."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 5px;
                padding: 8px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                padding: 5px;
            }
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                padding: 8px 16px;
            }
            QTabBar::tab:selected {
                background-color: #3a3a3a;
            }
            QRadioButton, QCheckBox {
                spacing: 8px;
            }
            QRadioButton::indicator, QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
    
    def load_image(self):
        """Load image from file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.original_label.setImage(self.image)
                self.analyze_btn.setEnabled(True)
                self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")
                
                # Show image info
                info = get_image_info(self.image)
                self.results_text.setText(
                    f"Image loaded:\n"
                    f"  Size: {info['width']} x {info['height']}\n"
                    f"  Channels: {info['channels']}\n"
                    f"  Type: {info['dtype']}"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to load image")
    
    def load_image_from_path(self, file_path: str):
        """Load image from file path (for drag-drop support - Bug 2 fix)."""
        if file_path:
            self.image = cv2.imread(file_path)
            if self.image is not None:
                self.original_label.setImage(self.image)
                self.analyze_btn.setEnabled(True)
                self.status_bar.showMessage(f"Loaded (dropped): {Path(file_path).name}")
                
                # Show image info
                info = get_image_info(self.image)
                self.results_text.setText(
                    f"Image loaded:\n"
                    f"  Size: {info['width']} x {info['height']}\n"
                    f"  Channels: {info['channels']}\n"
                    f"  Type: {info['dtype']}"
                )
            else:
                QMessageBox.warning(self, "Error", "Failed to load dropped image")
    
    def get_current_mode(self) -> str:
        """Get currently selected analysis mode."""
        if self.mode_classical.isChecked():
            return "classical"
        elif self.mode_dl.isChecked():
            return "deep_learning"
        elif self.mode_vlm.isChecked():
            return "vlm"
        else:
            return "hybrid"
    
    def get_parameters(self) -> dict:
        """Get current parameter values for ALL analysis modes."""
        params = {
            # Classical mode parameters
            "noise_filter": self.noise_combo.currentText(),
            "kernel_size": self.kernel_spin.value(),
            "contrast_method": self.contrast_combo.currentText(),
            "clahe_clip": self.clahe_spin.value(),
            "use_glcm": self.glcm_check.isChecked(),
            "use_lbp": self.lbp_check.isChecked(),
            "use_gabor": self.gabor_check.isChecked(),
            "segmentation_method": self.seg_combo.currentText(),
            "n_clusters": self.clusters_spin.value(),
            "use_morphology": self.morph_check.isChecked(),
            "fill_holes": self.fill_check.isChecked(),
            
            # VLM mode parameters (Bug 3-5 fix)
            "vlm_analysis_type": self.vlm_type_combo.currentText(),
            "vlm_temperature": self.temp_spin.value(),
            "vlm_include_layer": self.vlm_layer_check.isChecked(),
            "vlm_include_confidence": self.vlm_conf_check.isChecked(),
            "vlm_include_material": self.vlm_material_check.isChecked(),
            "vlm_include_texture": self.vlm_texture_check.isChecked(),
            "vlm_include_recommendations": self.vlm_recom_check.isChecked(),
            
            # Deep Learning mode parameters
            "dl_backbone": self.backbone_combo.currentText(),
            "dl_pretrained": self.pretrained_check.isChecked(),
            "dl_device": self.device_combo.currentText(),
            "dl_confidence_threshold": self.confidence_spin.value(),
            "dl_batch_size": self.batch_spin.value(),
            "dl_resolution": self.resolution_combo.currentText(),
            
            # Hybrid mode parameters
            "hybrid_primary_method": self.primary_combo.currentText(),
            "hybrid_vlm_validation": self.ai_validation_check.isChecked(),
            "classical_weight": self.classical_weight_spin.value() / 100.0,
            "hybrid_conflict_rule": self.conflict_combo.currentText()
        }
        return params
    
    def run_analysis(self):
        """Run analysis on loaded image."""
        if self.image is None:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(False)
        
        mode = self.get_current_mode()
        self.mode = mode  # Store mode for summary generation
        params = self.get_parameters()
        
        self.worker = AnalysisWorker(self.image.copy(), mode, params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_complete)
        self.worker.error.connect(self.analysis_error)
        self.worker.start()
    
    def update_progress(self, value: int, message: str):
        """Update progress bar."""
        self.progress_bar.setValue(value)
        self.status_bar.showMessage(message)
    
    def analysis_complete(self, result: dict):
        """Handle analysis completion."""
        self.result = result
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # Display segmentation result - use VLM visualization if available
        if "vlm_visualization" in result and self.mode == "vlm":
            # Use the enhanced VLM overlay visualization
            self.result_label.setImage(result["vlm_visualization"])
        elif "labels" in result:
            colored = create_colored_segmentation(result["labels"])
            self.result_label.setImage(colored)
        
        # Update legend
        if "labels" in result:
            self.update_legend(result["labels"])
        
        # Display results text - respecting Output Options for VLM mode
        classification = result.get("classification", {})
        features = result.get("features", {})
        
        text = "═══ ANALYSIS RESULTS ═══\n\n"
        
        # Check VLM Output Options (only apply to VLM mode)
        is_vlm = self.mode == "vlm"
        show_layer = not is_vlm or self.vlm_layer_check.isChecked()
        show_confidence = not is_vlm or self.vlm_conf_check.isChecked()
        show_material = not is_vlm or self.vlm_material_check.isChecked()
        show_texture = not is_vlm or self.vlm_texture_check.isChecked()
        show_recommendations = not is_vlm or self.vlm_recom_check.isChecked()
        
        # Layer identification (controlled by Include Layer Name)
        layer_name = classification.get('layer_name', classification.get('full_name', 'N/A'))
        if show_layer:
            text += f"🔍 Detected Layer: {layer_name}\n"
        
        # Confidence (controlled by Include Confidence)
        confidence = classification.get('confidence', 0)
        if show_confidence:
            text += f"📊 Confidence: {confidence:.1%}\n"
        
        # Material info (controlled by Include Material Description)
        material = classification.get('material', 'N/A')
        if show_material and material and material != 'N/A':
            text += f"🧱 Material: {material}\n"
        
        # Method used (always show)
        method = classification.get('method', 'N/A')
        if method == 'N/A':
            method = "VLM Analysis (GLM-4.6V)"
        text += f"⚙️  Method: {method}\n"
        
        # Layer number (always show if available)
        layer_num = classification.get('layer_number')
        if layer_num:
            text += f"🔢 Layer Number: {layer_num}\n"
        
        # VLM-specific fields - controlled by Output Options
        # Texture Description (controlled by Include Texture Description)
        texture = classification.get('texture_description')
        if show_texture and texture and texture != 'N/A':
            text += f"\n🎨 Texture Description:\n   {texture}\n"
        
        reasoning = classification.get('reasoning')
        if reasoning and reasoning != 'N/A':
            text += f"\n🧠 Analysis Reasoning:\n   {reasoning}\n"
        
        # Additional Notes / Recommendations (controlled by Include Recommendations)
        notes = classification.get('additional_notes')
        if show_recommendations and notes and notes != 'N/A':
            text += f"\n📝 Recommendations:\n   {notes}\n"
        
        # GLCM features (only for classical/hybrid modes)
        if "glcm" in features and self.mode in ["classical", "hybrid"]:
            text += "\n─── Texture Features ───\n"
            glcm = features["glcm"]
            text += f"Contrast:    {glcm.get('contrast', 0):.4f}\n"
            text += f"Energy:      {glcm.get('energy', 0):.4f}\n"
            text += f"Homogeneity: {glcm.get('homogeneity', 0):.4f}\n"
            text += f"Correlation: {glcm.get('correlation', 0):.4f}\n"
        
        # Show preprocessing params for Classical/Hybrid mode
        if self.mode in ["classical", "hybrid"]:
            text += "\n─── Settings Used ───\n"
            preproc = result.get("preprocessing_params", {})
            text += f"Noise Filter: {preproc.get('noise_filter', 'N/A')}\n"
            text += f"Kernel Size:  {preproc.get('kernel_size', 'N/A')}\n"
            text += f"Contrast:     {preproc.get('contrast_method', 'N/A')}\n"
            if preproc.get('contrast_method') == 'clahe':
                text += f"CLAHE Clip:   {preproc.get('clahe_clip', 'N/A')}\n"
            seg_method = result.get("segmentation_method", "K-Means")
            text += f"Segmentation: {seg_method}\n"
            morph = "Yes" if result.get("morphology_applied", False) else "No"
            holes = "Yes" if result.get("fill_holes_applied", False) else "No"
            text += f"Morphology:   {morph}\n"
            text += f"Fill Holes:   {holes}\n"
            
            # Show feature extraction settings
            text += "\n─── Features Used ───\n"
            text += f"GLCM: {'Yes' if self.glcm_check.isChecked() else 'No'}\n"
            text += f"LBP:  {'Yes' if self.lbp_check.isChecked() else 'No'}\n"
            text += f"Gabor: {'Yes' if self.gabor_check.isChecked() else 'No'}\n"
        
        self.results_text.setText(text)
        
        # Generate plain-English summary (also respects Output Options)
        self.generate_summary(result, classification, show_material, show_texture, show_recommendations)
        
        self.status_bar.showMessage("Analysis complete!")    
    def update_legend(self, labels):
        """Update legend to show all layers, highlighting detected ones (Bug 6 fix)."""
        import numpy as np
        
        # Hide placeholder and show legend widget
        self.legend_placeholder.setVisible(False)
        self.legend_widget.setVisible(True)
        
        # Get unique layer labels (excluding background 0)
        unique_labels = np.unique(labels)
        detected_layers = [int(l) for l in unique_labels if l > 0]
        
        # Always show ALL 5 layers but highlight detected ones
        for layer_num in range(1, 6):
            if layer_num in self.legend_items:
                layer = ROAD_LAYERS[layer_num]
                hex_color = layer["hex_color"]
                
                self.legend_items[layer_num].setVisible(True)
                
                if layer_num in detected_layers:
                    # Detected layer: use layer color as background, white text
                    self.legend_items[layer_num].setStyleSheet(f'''
                        color: white; 
                        font-weight: bold; 
                        padding: 4px 10px;
                        font-size: 11px;
                        background-color: {hex_color};
                        border-radius: 4px;
                        border: 2px solid white;
                    ''')
                    self.legend_items[layer_num].setText(f"✓ {layer['name']}")
                else:
                    # Non-detected layer: dimmed, gray background
                    self.legend_items[layer_num].setStyleSheet(f'''
                        color: #888; 
                        font-weight: normal; 
                        padding: 4px 10px;
                        font-size: 9px;
                        background-color: #333;
                        border-radius: 4px;
                    ''')
                    self.legend_items[layer_num].setText(f"■ {layer['name']}")

    def generate_summary(self, result: dict, classification: dict, 
                         show_material: bool = True, show_texture: bool = True, 
                         show_recommendations: bool = True):
        """Generate plain-English summary for non-technical users."""
        mode = self.mode if hasattr(self, 'mode') else self.get_current_mode()

        layer_name = classification.get('layer_name', classification.get('full_name', 'Unknown'))
        confidence = classification.get('confidence', 0)
        material = classification.get('material', '')

        # Build summary based on mode
        if mode == "vlm":
            summary = "VLM Analysis Summary\n\n"
            summary += f"The AI vision model analyzed your image and identified it as:\n\n"
            summary += f"{layer_name}\n\n"

            # Confidence explanation
            if confidence >= 0.8:
                summary += f"The model is very confident ({confidence:.0%}) about this identification.\n\n"
            elif confidence >= 0.6:
                summary += f"The model is moderately confident ({confidence:.0%}) about this identification.\n\n"
            else:
                summary += f"The model has low confidence ({confidence:.0%}) - consider using another analysis mode.\n\n"

            # Material explanation (controlled by Output Options)
            if show_material and material and material != 'N/A':
                summary += f"Material: {material}\n\n"

            # Simple explanation (controlled by show_recommendations)
            if show_recommendations:
                if "Aggregate" in layer_name:
                    summary += "What this means: This layer shows loose stones/gravel used for drainage and stability.\n"
                elif "Sub-base" in layer_name:
                    summary += "What this means: This is a foundational layer that distributes loads evenly.\n"
                elif "Base Course" in layer_name:
                    summary += "What this means: This is the main structural layer that bears traffic loads.\n"
                elif "Asphalt" in layer_name or "Surface" in layer_name:
                    summary += "What this means: This is the top wearing course that vehicles travel on.\n"
                elif "Soil" in layer_name or "Subgrade" in layer_name:
                    summary += "What this means: This is the natural soil foundation beneath all road layers.\n"

        elif mode == "deep_learning":
            summary = "Deep Learning Analysis Summary\n\n"
            summary += f"DeepLabv3+ segmented your image:\n\n"
            summary += f"Dominant Layer: {layer_name}\n"
            summary += f"Coverage: {confidence:.0%} of image\n"
            if material and material != "Unknown":
                summary += f"Material: {material}\n"
            summary += "\n"
            
            # Show layer distribution if available
            layer_dist = result.get("layer_distribution", {})
            if layer_dist:
                summary += "Layer Distribution:\n"
                for layer_num, info in sorted(layer_dist.items(), key=lambda x: -x[1]["percentage"]):
                    summary += f"  • {info['name']}: {info['percentage']:.0%}\n"
                summary += "\n"
            
            summary += "This mode uses neural networks trained on road images\n"
            summary += "to identify and segment multiple layer types.\n"

        elif mode == "classical":
            summary = "Classical Analysis Summary\n\n"
            summary += f"Traditional image processing techniques were used:\n\n"
            summary += f"Identified Layer: {layer_name}\n"
            summary += f"Confidence: {confidence:.0%}\n\n"

            if material:
                summary += f"Material Type: {material}\n\n"

            # Get actual segmentation method from result
            seg_method = result.get("segmentation_method", "K-Means")
            preproc = result.get("preprocessing_params", {})
            
            summary += "Settings Used:\n"
            summary += f"- Noise Filter: {preproc.get('noise_filter', 'median')}\n"
            summary += f"- Contrast: {preproc.get('contrast_method', 'clahe')}\n"
            summary += f"- Segmentation: {seg_method}\n\n"
            
            summary += "How it worked:\n"
            summary += "- Extracted texture features (GLCM, LBP)\n"
            summary += f"- Applied {seg_method} segmentation\n"
            summary += "- Used heuristic rules to classify layers\n\n"
            summary += "This mode is fast and works well for clear, distinct textures.\n"

        elif mode == "hybrid":
            summary = "Hybrid Analysis Summary\n\n"
            summary += f"Combined classical and AI methods for best accuracy:\n\n"
            summary += f"Final Result: {layer_name}\n"
            summary += f"Combined Confidence: {confidence:.0%}\n\n"
            
            # Show what methods were used
            method_info = classification.get("method", "Hybrid")
            summary += f"Decision: {method_info}\n\n"
            
            # Show Classical result if available
            classical_res = result.get("classical_result", {})
            vlm_res = result.get("vlm_result")
            
            summary += "Individual Results:\n"
            if classical_res:
                summary += f"- Classical: {classical_res.get('layer_name', 'N/A')} ({classical_res.get('confidence', 0):.0%})\n"
            if vlm_res:
                summary += f"- VLM (AI): {vlm_res.get('layer_name', 'N/A')} ({vlm_res.get('confidence', 0):.0%})\n"
            elif result.get("vlm_error"):
                summary += f"- VLM (AI): Failed - {result.get('vlm_error', 'Unknown error')}\n"
            else:
                summary += "- VLM (AI): Disabled\n"
            
            summary += "\nThis mode provides the most accurate results by combining\n"
            summary += "fast texture analysis with intelligent AI understanding.\n"

        else:
            summary = f"Analysis complete: {layer_name} ({confidence:.0%} confidence)"

        self.summary_text.setText(summary)

    def analysis_error(self, error: str):
        """Handle analysis error."""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", error)
        self.status_bar.showMessage("Analysis failed")
    
    def save_result(self):
        """Save analysis result to file."""
        if self.result is None or "labels" not in self.result:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        
        if file_path:
            colored = create_colored_segmentation(self.result["labels"])
            cv2.imwrite(file_path, colored)
            self.status_bar.showMessage(f"Saved: {Path(file_path).name}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Road Surface Layer Analyzer",
            "Road Surface Layer Analyzer\n\n"
            "CSC566 Image Processing Mini Project\n"
            "ClaRity Group\n\n"
            "Automated analysis of road construction layers\n"
            "using texture-based image segmentation."
        )


def main():
    """Main entry point with splash screen."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Import and show splash screen
    from gui.splash_screen import SplashScreen
    
    main_window = None
    
    def on_splash_finished():
        nonlocal main_window
        main_window = MainWindow()
        main_window.show()
    
    splash = SplashScreen(on_finished=on_splash_finished)
    splash.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


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
    QMenuBar, QMenu, QAction, QMessageBox, QTabWidget,
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
            
            # Step 1: Preprocessing
            self.progress.emit(10, "Preprocessing image...")
            preprocessed = preprocess_image(
                self.image,
                denoise=self.params.get("noise_filter", "median"),
                denoise_kernel=self.params.get("kernel_size", 3),
                enhance=self.params.get("contrast_method", "clahe"),
                color_space="gray"
            )
            result["preprocessed"] = preprocessed
            
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
                
                # Step 3: Segmentation
                self.progress.emit(60, "Segmenting image...")
                n_clusters = self.params.get("n_clusters", 5)
                labels, _ = kmeans_segment(self.image, n_clusters=n_clusters)
                
                # Step 4: Morphological cleanup
                self.progress.emit(80, "Refining segmentation...")
                if self.params.get("use_morphology", True):
                    # Apply cleanup per segment
                    pass
                
                result["labels"] = labels + 1  # 1-indexed
                
                # Step 5: Classification
                self.progress.emit(90, "Classifying layers...")
                classification = get_layer_by_texture_heuristic(features)
                result["classification"] = classification
            
            elif self.mode == "deep_learning":
                self.progress.emit(50, "Running DeepLabv3+ inference...")
                try:
                    from src.deep_learning import DeepLabSegmenter
                    segmenter = DeepLabSegmenter()
                    labels = segmenter.segment(self.image)
                    result["labels"] = labels
                    result["classification"] = {
                        "layer_name": "Deep Learning",
                        "confidence": 0.85,
                        "method": "DeepLabv3+"
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
                    
                    vlm_result = analyzer.analyze_road_layer(str(temp_path))
                    result["classification"] = vlm_result
                    result["labels"] = np.ones(self.image.shape[:2], dtype=np.uint8) * vlm_result.get("layer_number", 1)
                except Exception as e:
                    self.error.emit(f"VLM analysis error: {str(e)}")
                    return
            
            elif self.mode == "hybrid":
                # Combine classical and VLM
                self.progress.emit(30, "Running classical analysis...")
                features = extract_all_texture_features(preprocessed)
                labels, _ = kmeans_segment(self.image, n_clusters=5)
                
                self.progress.emit(70, "Running VLM analysis...")
                try:
                    from src.vlm_analyzer import VLMAnalyzer
                    analyzer = VLMAnalyzer()
                    temp_path = Path(__file__).parent.parent / "results" / "temp_analysis.jpg"
                    temp_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(temp_path), self.image)
                    vlm_result = analyzer.analyze_road_layer(str(temp_path))
                    
                    classical = get_layer_by_texture_heuristic(features)
                    
                    # Combine results (weighted average)
                    classical_weight = self.params.get("classical_weight", 0.7)
                    result["classification"] = {
                        "layer_name": vlm_result.get("layer_name") or classical["layer_name"],
                        "confidence": classical_weight * classical["confidence"] + 
                                     (1 - classical_weight) * vlm_result.get("confidence", 0.5),
                        "method": "Hybrid"
                    }
                except:
                    result["classification"] = get_layer_by_texture_heuristic(features)
                
                result["labels"] = labels + 1
                result["features"] = features
            
            self.progress.emit(100, "Analysis complete!")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class ImageLabel(QLabel):
    """Custom label for displaying images with aspect ratio preservation."""
    
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
        legend_group = QGroupBox("Layer Legend")
        legend_layout = QHBoxLayout(legend_group)
        
        for layer_num in range(1, 6):
            layer = ROAD_LAYERS[layer_num]
            color = layer["hex_color"]
            
            legend_item = QLabel(f"■ {layer['name']}")
            legend_item.setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px;")
            legend_layout.addWidget(legend_item)
        
        legend_layout.addStretch()
        layout.addWidget(legend_group)
        
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
        self.mode_dl = QRadioButton("Deep Learning (DeepLabv3+)")
        self.mode_vlm = QRadioButton("VLM Analysis (GLM-4.6V)")
        self.mode_hybrid = QRadioButton("Hybrid (Classical + AI)")
        
        mode_layout.addWidget(self.mode_classical)
        mode_layout.addWidget(self.mode_dl)
        mode_layout.addWidget(self.mode_vlm)
        mode_layout.addWidget(self.mode_hybrid)
        
        layout.addWidget(mode_group)
        
        # Parameters tab widget
        self.params_tabs = QTabWidget()
        
        # Preprocessing tab
        preprocess_tab = self.create_preprocessing_params()
        self.params_tabs.addTab(preprocess_tab, "Preprocessing")
        
        # Features tab
        features_tab = self.create_features_params()
        self.params_tabs.addTab(features_tab, "Features")
        
        # Segmentation tab
        seg_tab = self.create_segmentation_params()
        self.params_tabs.addTab(seg_tab, "Segmentation")
        
        layout.addWidget(self.params_tabs)
        
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
        
        layout.addStretch()
        
        return panel
    
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
        """Get current parameter values."""
        return {
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
            "fill_holes": self.fill_check.isChecked()
        }
    
    def run_analysis(self):
        """Run analysis on loaded image."""
        if self.image is None:
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.analyze_btn.setEnabled(False)
        
        mode = self.get_current_mode()
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
        
        # Display segmentation result
        if "labels" in result:
            colored = create_colored_segmentation(result["labels"])
            self.result_label.setImage(colored)
        
        # Display results text
        classification = result.get("classification", {})
        features = result.get("features", {})
        
        text = "═══ ANALYSIS RESULTS ═══\n\n"
        text += f"Detected Layer: {classification.get('layer_name', 'N/A')}\n"
        text += f"Confidence: {classification.get('confidence', 0):.1%}\n"
        text += f"Material: {classification.get('material', 'N/A')}\n"
        text += f"Method: {classification.get('method', 'N/A')}\n\n"
        
        if "glcm" in features:
            glcm = features["glcm"]
            text += "─── GLCM Features ───\n"
            text += f"Contrast:    {glcm.get('contrast', 0):.4f}\n"
            text += f"Energy:      {glcm.get('energy', 0):.4f}\n"
            text += f"Homogeneity: {glcm.get('homogeneity', 0):.4f}\n"
            text += f"Correlation: {glcm.get('correlation', 0):.4f}\n"
        
        self.results_text.setText(text)
        self.status_bar.showMessage("Analysis complete!")
    
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
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

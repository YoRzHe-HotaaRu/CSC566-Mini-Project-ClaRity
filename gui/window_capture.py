"""
Window Capture Module for Live Preview
Handles window enumeration, thumbnail generation, and real-time screen capture.

CSC566 Image Processing Mini Project
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from threading import Thread, Event
from queue import Queue

import cv2
import mss
import win32gui
import win32con
import win32ui
import win32api
from ctypes import windll

from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QScrollArea, QWidget, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


@dataclass
class WindowInfo:
    """Information about a window."""
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]  # left, top, right, bottom
    thumbnail: Optional[np.ndarray] = None


class WindowEnumerator:
    """Enumerate visible windows and generate thumbnails."""
    
    @staticmethod
    def get_visible_windows() -> List[WindowInfo]:
        """Get list of all visible windows with titles."""
        windows = []
        
        def enum_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and len(title) > 0:
                    try:
                        rect = win32gui.GetWindowRect(hwnd)
                        # Filter out tiny/hidden windows
                        width = rect[2] - rect[0]
                        height = rect[3] - rect[1]
                        if width > 100 and height > 100:
                            windows.append(WindowInfo(
                                hwnd=hwnd,
                                title=title,
                                rect=rect
                            ))
                    except Exception:
                        pass
            return True
        
        win32gui.EnumWindows(enum_callback, windows)
        return windows
    
    @staticmethod
    def capture_window_thumbnail(hwnd: int, max_size: int = 200) -> Optional[np.ndarray]:
        """Capture a thumbnail of the specified window."""
        try:
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top
            
            if width <= 0 or height <= 0:
                return None
            
            # Use mss for faster capture
            with mss.mss() as sct:
                monitor = {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height
                }
                screenshot = sct.grab(monitor)
                img = np.array(screenshot)
                # Convert BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Resize to thumbnail
            aspect = width / height
            if width > height:
                new_width = min(width, max_size)
                new_height = int(new_width / aspect)
            else:
                new_height = min(height, max_size)
                new_width = int(new_height * aspect)
            
            thumbnail = cv2.resize(img, (new_width, new_height))
            return thumbnail
            
        except Exception as e:
            print(f"Error capturing thumbnail: {e}")
            return None


class WindowSelectorDialog(QDialog):
    """Dialog for selecting a window to capture with visual thumbnails."""
    
    window_selected = pyqtSignal(object)  # Emits WindowInfo
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Window to Capture")
        self.setMinimumSize(700, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #2a2a4a;
                color: white;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #3a3a5a;
                border-color: #4CAF50;
            }
        """)
        
        self.selected_window = None
        self.init_ui()
        self.load_windows()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📺 Select a Window to Capture")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Scroll area for window grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(15)
        
        scroll.setWidget(self.grid_widget)
        layout.addWidget(scroll)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_windows)
        btn_layout.addWidget(refresh_btn)
        
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def load_windows(self):
        """Load and display all visible windows."""
        # Clear existing
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get windows
        windows = WindowEnumerator.get_visible_windows()
        
        # Create thumbnails for each window
        row, col = 0, 0
        max_cols = 3
        
        for window in windows:
            # Skip our own window
            if "Select Window" in window.title:
                continue
            
            # Create thumbnail
            thumbnail = WindowEnumerator.capture_window_thumbnail(window.hwnd)
            window.thumbnail = thumbnail
            
            # Create card widget
            card = self._create_window_card(window)
            self.grid_layout.addWidget(card, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
    
    def _create_window_card(self, window: WindowInfo) -> QFrame:
        """Create a clickable card for a window."""
        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel)
        card.setStyleSheet("""
            QFrame {
                background-color: #2a2a4a;
                border: 2px solid #444;
                border-radius: 8px;
                padding: 5px;
            }
            QFrame:hover {
                border-color: #4CAF50;
            }
        """)
        card.setFixedSize(200, 180)
        card.setCursor(Qt.PointingHandCursor)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(5)
        
        # Thumbnail
        thumb_label = QLabel()
        thumb_label.setAlignment(Qt.AlignCenter)
        thumb_label.setFixedHeight(120)
        
        if window.thumbnail is not None:
            # Convert numpy to QPixmap
            h, w = window.thumbnail.shape[:2]
            bytes_per_line = 3 * w
            qimg = QImage(window.thumbnail.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimg)
            thumb_label.setPixmap(pixmap.scaled(180, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb_label.setText("📷 No preview")
            thumb_label.setStyleSheet("color: #888;")
        
        layout.addWidget(thumb_label)
        
        # Title (truncated)
        title_text = window.title[:30] + "..." if len(window.title) > 30 else window.title
        title_label = QLabel(title_text)
        title_label.setStyleSheet("font-size: 10px; color: #ccc;")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        layout.addWidget(title_label)
        
        # Make card clickable
        card.mousePressEvent = lambda event, w=window: self._select_window(w)
        
        return card
    
    def _select_window(self, window: WindowInfo):
        """Handle window selection."""
        self.selected_window = window
        self.window_selected.emit(window)
        self.accept()


class WindowCaptureThread(QThread):
    """Thread for continuous window capture at target FPS using PrintWindow."""
    
    frame_ready = pyqtSignal(np.ndarray)  # Emits captured frame
    fps_updated = pyqtSignal(float)  # Emits current FPS
    error = pyqtSignal(str)
    
    def __init__(self, hwnd: int, target_fps: int = 22):
        super().__init__()
        self.hwnd = hwnd
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.running = False
        self.paused = False
        self.skip_frames = 0  # Skip frames when processing is slow
    
    def _capture_window_content(self) -> Optional[np.ndarray]:
        """Capture window content using PrintWindow (no overlapping windows)."""
        try:
            # Get window dimensions
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            width = right - left
            height = bottom - top
            
            if width <= 0 or height <= 0:
                return None
            
            # Get device context
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)
            
            # Use PrintWindow to capture only window content (not screen region)
            # Flag 2 = PW_RENDERFULLCONTENT (captures even if window is occluded)
            result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 2)
            
            if result == 0:
                # Fallback: use BitBlt if PrintWindow fails
                saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
            
            # Convert to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img = img.reshape((height, width, 4))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Clean up
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)
            
            return img
            
        except Exception as e:
            return None
    
    def run(self):
        """Main capture loop."""
        self.running = True
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            frame_start = time.time()
            
            try:
                # Check window exists
                if not win32gui.IsWindow(self.hwnd):
                    self.error.emit("Window no longer exists")
                    break
                
                # Skip frames if needed (performance optimization)
                if self.skip_frames > 0:
                    self.skip_frames -= 1
                    time.sleep(self.frame_interval)
                    continue
                
                # Capture using PrintWindow
                frame = self._capture_window_content()
                
                if frame is not None:
                    self.frame_ready.emit(frame)
                    frame_count += 1
                
                # Update FPS every second
                current_time = time.time()
                if current_time - last_fps_update >= 1.0:
                    fps = frame_count / (current_time - last_fps_update)
                    self.fps_updated.emit(fps)
                    frame_count = 0
                    last_fps_update = current_time
                
                # Maintain target FPS
                elapsed = time.time() - frame_start
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.error.emit(str(e))
                time.sleep(0.5)
    
    def stop(self):
        """Stop the capture thread."""
        self.running = False
        self.wait(2000)  # Wait up to 2 seconds for thread to finish
    
    def pause(self):
        """Pause capture."""
        self.paused = True
    
    def resume(self):
        """Resume capture."""
        self.paused = False


class YOLOInferenceWorker(QThread):
    """Worker thread for async YOLO inference to prevent UI lag."""
    
    result_ready = pyqtSignal(np.ndarray, dict)  # Emits (visualization, prediction)
    
    def __init__(self, yolo_analyzer):
        super().__init__()
        self.yolo_analyzer = yolo_analyzer
        self.running = True
        self.frame_queue = Queue(maxsize=2)  # Small queue to avoid memory buildup
        self.preprocess_opts = {}
        self.display_opts = {
            "show_masks": True,
            "show_labels": True,
            "show_confidence": True,
            "mask_opacity": 0.4
        }
    
    def set_options(self, preprocess: dict, display: dict):
        """Update processing options."""
        self.preprocess_opts = preprocess
        self.display_opts = display
    
    def process_frame(self, frame: np.ndarray):
        """Add frame to processing queue (drops oldest if full)."""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()  # Drop oldest frame
            except:
                pass
        try:
            self.frame_queue.put_nowait(frame)
        except:
            pass
    
    def run(self):
        """Main processing loop."""
        while self.running:
            try:
                # Wait for frame with timeout
                frame = self.frame_queue.get(timeout=0.1)
                
                # Run YOLO inference
                prediction = self.yolo_analyzer.predict(frame, self.preprocess_opts)
                
                # Create visualization
                vis_frame = self.yolo_analyzer.visualize(
                    prediction.get("preprocessed", frame),
                    prediction,
                    show_masks=self.display_opts.get("show_masks", True),
                    show_labels=self.display_opts.get("show_labels", True),
                    show_confidence=self.display_opts.get("show_confidence", True),
                    mask_opacity=self.display_opts.get("mask_opacity", 0.4)
                )
                
                # Emit result
                self.result_ready.emit(vis_frame, prediction)
                
            except Exception as e:
                # Queue timeout or processing error - continue
                pass
    
    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.wait(2000)


class LivePreviewController:
    """Controller for live preview with YOLO inference."""
    
    def __init__(self, yolo_analyzer, display_callback: Callable[[np.ndarray], None]):
        """
        Initialize controller.
        
        Args:
            yolo_analyzer: YOLOAnalyzer instance
            display_callback: Function to call with processed frame
        """
        self.yolo_analyzer = yolo_analyzer
        self.display_callback = display_callback
        self.capture_thread: Optional[WindowCaptureThread] = None
        self.current_window: Optional[WindowInfo] = None
        self.processing = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_prediction: Optional[dict] = None
        
        # Display options (can be updated)
        self.show_masks = True
        self.show_labels = True
        self.show_confidence = True
        self.mask_opacity = 0.4
        self.preprocess_opts = {}
    
    def start_capture(self, window: WindowInfo, target_fps: int = 30):
        """Start capturing from the selected window."""
        self.current_window = window
        
        # Create and start capture thread
        self.capture_thread = WindowCaptureThread(window.hwnd, target_fps)
        self.capture_thread.frame_ready.connect(self._on_frame_received)
        self.capture_thread.start()
    
    def stop_capture(self):
        """Stop capturing."""
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
    
    def _on_frame_received(self, frame: np.ndarray):
        """Process received frame with YOLO."""
        if self.processing:
            return  # Skip if still processing previous frame
        
        self.processing = True
        self.last_frame = frame.copy()
        
        try:
            # Run YOLO inference
            prediction = self.yolo_analyzer.predict(frame, self.preprocess_opts)
            self.last_prediction = prediction
            
            # Create visualization
            vis_frame = self.yolo_analyzer.visualize(
                frame,
                prediction,
                show_masks=self.show_masks,
                show_labels=self.show_labels,
                show_confidence=self.show_confidence,
                mask_opacity=self.mask_opacity
            )
            
            # Send to display
            self.display_callback(vis_frame)
            
        except Exception as e:
            print(f"YOLO inference error: {e}")
            self.display_callback(frame)  # Show original on error
        
        finally:
            self.processing = False
    
    def capture_current_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
        """Capture current frame for analysis.
        
        Returns:
            Tuple of (original_frame, visualization, prediction) or None
        """
        if self.last_frame is not None and self.last_prediction is not None:
            vis = self.yolo_analyzer.visualize(
                self.last_frame,
                self.last_prediction,
                show_masks=self.show_masks,
                show_labels=self.show_labels,
                show_confidence=self.show_confidence,
                mask_opacity=self.mask_opacity
            )
            return (self.last_frame.copy(), vis, self.last_prediction)
        return None


class LiveAnalysisDialog(QDialog):
    """Dialog showing before/after comparison of a captured live frame."""
    
    def __init__(self, original: np.ndarray, visualization: np.ndarray, 
                 prediction: dict, params: dict, parent=None):
        super().__init__(parent)
        self.original = original
        self.visualization = visualization
        self.prediction = prediction
        self.params = params
        
        self.setWindowTitle("📊 Live Frame Analysis")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #2a2a4a;
                color: white;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #3a3a5a;
                border-color: #4CAF50;
            }
            QTextEdit {
                background-color: #2a2a3a;
                color: white;
                border: 1px solid #444;
                border-radius: 5px;
            }
        """)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🎯 YOLOv11 Live Frame Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)
        
        # Images side by side
        images_layout = QHBoxLayout()
        
        # Original image
        orig_group = QFrame()
        orig_group.setStyleSheet("QFrame { border: 2px solid #444; border-radius: 8px; padding: 5px; }")
        orig_layout = QVBoxLayout(orig_group)
        orig_title = QLabel("📷 Original Frame")
        orig_title.setStyleSheet("font-weight: bold;")
        orig_layout.addWidget(orig_title)
        
        self.orig_label = QLabel()
        self.orig_label.setAlignment(Qt.AlignCenter)
        self._set_image(self.orig_label, self.original, max_size=450)
        orig_layout.addWidget(self.orig_label)
        images_layout.addWidget(orig_group)
        
        # Segmented image
        seg_group = QFrame()
        seg_group.setStyleSheet("QFrame { border: 2px solid #4CAF50; border-radius: 8px; padding: 5px; }")
        seg_layout = QVBoxLayout(seg_group)
        seg_title = QLabel("🎯 YOLO Segmentation")
        seg_title.setStyleSheet("font-weight: bold; color: #4CAF50;")
        seg_layout.addWidget(seg_title)
        
        self.seg_label = QLabel()
        self.seg_label.setAlignment(Qt.AlignCenter)
        self._set_image(self.seg_label, self.visualization, max_size=450)
        seg_layout.addWidget(self.seg_label)
        images_layout.addWidget(seg_group)
        
        layout.addLayout(images_layout)
        
        # Detection details
        from PyQt5.QtWidgets import QTextEdit
        details_group = QFrame()
        details_group.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 5px; padding: 5px; }")
        details_layout = QVBoxLayout(details_group)
        details_title = QLabel("📋 Detection Details")
        details_title.setStyleSheet("font-weight: bold;")
        details_layout.addWidget(details_title)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(150)
        self._populate_details()
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        export_btn = QPushButton("📄 Export PDF Report")
        export_btn.setStyleSheet("background-color: #4CAF50;")
        export_btn.clicked.connect(self.export_pdf)
        btn_layout.addWidget(export_btn)
        
        save_img_btn = QPushButton("💾 Save Images")
        save_img_btn.clicked.connect(self.save_images)
        btn_layout.addWidget(save_img_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _set_image(self, label: QLabel, image: np.ndarray, max_size: int = 400):
        """Set numpy image to QLabel."""
        h, w = image.shape[:2]
        aspect = w / h
        if w > h:
            new_w = min(w, max_size)
            new_h = int(new_w / aspect)
        else:
            new_h = min(h, max_size)
            new_w = int(new_h * aspect)
        
        resized = cv2.resize(image, (new_w, new_h))
        bytes_per_line = 3 * new_w
        qimg = QImage(resized.data, new_w, new_h, bytes_per_line, QImage.Format_BGR888)
        label.setPixmap(QPixmap.fromImage(qimg))
    
    def _populate_details(self):
        """Populate detection details text."""
        from src.config import ROAD_LAYERS
        
        detections = self.prediction.get("detections", [])
        text = f"Total Detections: {len(detections)}\n\n"
        
        # Group by layer
        layer_groups = {}
        for det in detections:
            layer_num = det["layer_number"]
            if layer_num not in layer_groups:
                layer_groups[layer_num] = []
            layer_groups[layer_num].append(det)
        
        for layer_num in sorted(layer_groups.keys()):
            layer_dets = layer_groups[layer_num]
            layer_info = ROAD_LAYERS.get(layer_num, {})
            layer_name = layer_info.get("name", f"Layer {layer_num}")
            avg_conf = sum(d["confidence"] for d in layer_dets) / len(layer_dets)
            
            text += f"🔷 L{layer_num}: {layer_name}\n"
            text += f"   Instances: {len(layer_dets)}, Avg Confidence: {avg_conf:.1%}\n\n"
        
        self.details_text.setText(text)
    
    def export_pdf(self):
        """Export analysis to PDF report."""
        from PyQt5.QtWidgets import QFileDialog
        from datetime import datetime
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report",
            f"live_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf)"
        )
        
        if filename:
            try:
                from src.report_generator import ReportGenerator
                
                # Prepare result dict
                result = {
                    "labels": self.prediction.get("masks"),
                    "detections": self.prediction.get("detections"),
                    "colored_segmentation": self.visualization,
                    "classification": {
                        "layer_number": self.prediction.get("dominant_layer"),
                        "layer_name": "Multiple Layers",
                        "confidence": self.prediction.get("dominant_confidence", 0),
                        "method": "YOLOv11 Live Capture"
                    },
                    "source_filename": "Live Capture",
                    "processing_time": 0
                }
                
                generator = ReportGenerator(
                    self.original.copy(),
                    result,
                    self.params,
                    "yolo"
                )
                generator.generate(filename)
                
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", f"PDF saved to:\n{filename}")
                
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to export PDF:\n{str(e)}")
    
    def save_images(self):
        """Save original and segmented images."""
        from PyQt5.QtWidgets import QFileDialog
        from datetime import datetime
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Images",
            f"live_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png)"
        )
        
        if filename:
            try:
                base = filename.rsplit('.', 1)[0]
                cv2.imwrite(f"{base}_original.png", self.original)
                cv2.imwrite(f"{base}_segmented.png", self.visualization)
                
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.information(self, "Success", 
                    f"Images saved:\n{base}_original.png\n{base}_segmented.png")
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", f"Failed to save images:\n{str(e)}")


if __name__ == "__main__":
    # Test window enumeration
    print("Enumerating windows...")
    windows = WindowEnumerator.get_visible_windows()
    for w in windows[:10]:
        print(f"  - {w.title[:50]}")

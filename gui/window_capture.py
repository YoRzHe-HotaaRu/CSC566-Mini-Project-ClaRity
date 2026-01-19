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
    """Thread for continuous window capture at target FPS."""
    
    frame_ready = pyqtSignal(np.ndarray)  # Emits captured frame
    fps_updated = pyqtSignal(float)  # Emits current FPS
    error = pyqtSignal(str)
    
    def __init__(self, hwnd: int, target_fps: int = 30):
        super().__init__()
        self.hwnd = hwnd
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.running = False
        self.paused = False
    
    def run(self):
        """Main capture loop."""
        self.running = True
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
        
        with mss.mss() as sct:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                frame_start = time.time()
                
                try:
                    # Get window rect
                    if not win32gui.IsWindow(self.hwnd):
                        self.error.emit("Window no longer exists")
                        break
                    
                    left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
                    width = right - left
                    height = bottom - top
                    
                    if width <= 0 or height <= 0:
                        time.sleep(0.1)
                        continue
                    
                    # Capture
                    monitor = {
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height
                    }
                    screenshot = sct.grab(monitor)
                    frame = np.array(screenshot)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    
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
        self.wait(timeout=2000)
    
    def pause(self):
        """Pause capture."""
        self.paused = True
    
    def resume(self):
        """Resume capture."""
        self.paused = False


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


if __name__ == "__main__":
    # Test window enumeration
    print("Enumerating windows...")
    windows = WindowEnumerator.get_visible_windows()
    for w in windows[:10]:
        print(f"  - {w.title[:50]}")

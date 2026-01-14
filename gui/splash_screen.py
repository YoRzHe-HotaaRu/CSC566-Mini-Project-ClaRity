"""
Splash Screen for Road Surface Layer Analyzer
Shows welcome screen with logo, title, and loading animation.

CSC566 Image Processing Mini Project
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGraphicsOpacityEffect, QApplication
)
from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, 
    QParallelAnimationGroup, QSequentialAnimationGroup
)
from PyQt5.QtGui import QPixmap, QFont, QPainter, QLinearGradient, QColor, QPalette


class SplashScreen(QWidget):
    """Animated splash screen with gradient background."""
    
    VERSION = "1.0.0"
    
    def __init__(self, on_finished=None):
        super().__init__()
        self.on_finished = on_finished
        self.init_ui()
        self.setup_animations()
        
    def init_ui(self):
        """Initialize the splash screen UI."""
        # Window settings
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(600, 500)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(15)
        
        # Spacer at top
        layout.addStretch(1)
        
        # Logo
        self.logo_label = QLabel()
        logo_path = Path(__file__).parent / "assets" / "logo.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            scaled = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.logo_label.setPixmap(scaled)
        else:
            self.logo_label.setText("🛣️")
            self.logo_label.setFont(QFont("Segoe UI", 48))
        self.logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo_label)
        
        # Title
        self.title_label = QLabel("Road Surface Layer Analyzer")
        self.title_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.title_label.setStyleSheet("color: white;")
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Subtitle / Slogan
        self.slogan_label = QLabel("Building the future with ClaRity")
        self.slogan_label.setFont(QFont("Segoe UI", 12, QFont.StyleItalic))
        self.slogan_label.setStyleSheet("color: #88ccaa;")
        self.slogan_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.slogan_label)
        
        # Spacer
        layout.addSpacing(20)
        
        # Course info
        self.course_label = QLabel("CSC566 Image Processing Mini Project")
        self.course_label.setFont(QFont("Segoe UI", 10))
        self.course_label.setStyleSheet("color: #aabbcc;")
        self.course_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.course_label)
        
        # Version
        self.version_label = QLabel(f"Version {self.VERSION}")
        self.version_label.setFont(QFont("Segoe UI", 9))
        self.version_label.setStyleSheet("color: #888;")
        self.version_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.version_label)
        
        # Spacer
        layout.addSpacing(30)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2a3a4a;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #2196F3);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Loading text - bigger with animated messages
        self.loading_label = QLabel("")
        self.loading_label.setFont(QFont("Segoe UI", 14))
        self.loading_label.setStyleSheet("color: #aaddbb;")
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)
        
        # Loading messages for typewriter effect
        self.loading_messages = [
            "Initializing components...",
            "Loading texture analyzers...",
            "Preparing segmentation engine...",
            "Connecting to AI systems...",
            "Almost ready...",
            "Welcome!"
        ]
        self.current_message_idx = 0
        self.current_char_idx = 0
        
        # Spacer
        layout.addStretch(2)
        
        # Made by text
        self.credits_label = QLabel("Made by ClaRity Group")
        self.credits_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.credits_label.setStyleSheet("color: #aaa;")
        self.credits_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.credits_label)
        
    def paintEvent(self, event):
        """Paint gradient background."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create gradient: soft blue -> black -> soft green
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(30, 50, 70))      # Soft blue-gray
        gradient.setColorAt(0.5, QColor(20, 25, 30))      # Dark center
        gradient.setColorAt(1.0, QColor(30, 50, 45))      # Soft green-gray
        
        painter.fillRect(self.rect(), gradient)
        
        # Add subtle border
        painter.setPen(QColor(60, 80, 100))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 10, 10)
        
    def setup_animations(self):
        """Setup loading animation and auto-dismiss timer."""
        # Progress animation
        self.progress_value = 0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(70)  # Update every 70ms for 7 seconds total
        
        # Auto-dismiss timer (7 seconds)
        self.auto_dismiss_timer = QTimer()
        self.auto_dismiss_timer.setSingleShot(True)
        self.auto_dismiss_timer.timeout.connect(self.finish)
        self.auto_dismiss_timer.start(7000)
        
        # Typewriter text animation
        self.typewriter_timer = QTimer()
        self.typewriter_timer.timeout.connect(self.animate_typewriter)
        self.typewriter_timer.start(50)  # Type every 50ms
        
    def update_progress(self):
        """Update progress bar."""
        self.progress_value += 1
        if self.progress_value <= 100:
            self.progress_bar.setValue(self.progress_value)
        else:
            self.progress_timer.stop()
            
    def animate_typewriter(self):
        """Typewriter animation for loading messages."""
        if self.current_message_idx >= len(self.loading_messages):
            self.typewriter_timer.stop()
            return
            
        current_msg = self.loading_messages[self.current_message_idx]
        
        if self.current_char_idx < len(current_msg):
            # Type next character
            self.loading_label.setText(current_msg[:self.current_char_idx + 1])
            self.current_char_idx += 1
        else:
            # Message complete, pause then move to next
            self.current_char_idx = 0
            self.current_message_idx += 1
            # Brief pause between messages
            self.typewriter_timer.stop()
            QTimer.singleShot(800, self.resume_typewriter)
    
    def resume_typewriter(self):
        """Resume typewriter after pause."""
        if self.current_message_idx < len(self.loading_messages):
            self.typewriter_timer.start(50)
        
    def finish(self):
        """Close splash and show main window."""
        self.progress_timer.stop()
        self.auto_dismiss_timer.stop()
        self.typewriter_timer.stop()
        
        if self.on_finished:
            self.on_finished()
        self.close()
        
    def keyPressEvent(self, event):
        """Handle Enter key press."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.finish()


def show_splash_then_main():
    """Show splash screen, then launch main window."""
    from gui.main_window import MainWindow
    
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    main_window = None
    
    def on_splash_finished():
        nonlocal main_window
        main_window = MainWindow()
        main_window.show()
    
    splash = SplashScreen(on_finished=on_splash_finished)
    splash.show()
    
    return app, splash


if __name__ == "__main__":
    app, splash = show_splash_then_main()
    sys.exit(app.exec_())

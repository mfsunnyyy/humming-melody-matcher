# gui/widgets.py
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QFont, QMovie


class RecordingIndicator(QWidget):
    """Pulsing red dot + text to show recording is active."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.is_recording = False
        self.pulse = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_pulse)
        self.timer.start(100)

    def set_recording(self, active: bool):
        self.is_recording = active
        self.pulse = 1.0 if active else 0.0
        self.update()

    def update_pulse(self):
        if self.is_recording:
            import time
            self.pulse = 0.5 + 0.5 * np.sin(time.time() * 10)
            self.update()
        else:
            self.pulse = 0.0
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor("#0b1322"))

        # Draw pulsing red dot
        dot_radius = int(15 * (0.8 + 0.2 * self.pulse))
        center_x = 40
        center_y = rect.height() // 2
        painter.setBrush(QColor("#e57373"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center_x - dot_radius, center_y - dot_radius, dot_radius * 2, dot_radius * 2)

        # Draw text
        painter.setPen(QColor("#e8f2ff"))
        painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
        if self.is_recording:
            painter.drawText(80, 35, "🔴 RECORDING...")
        else:
            painter.drawText(80, 35, "Ready")


class DancingCharacter(QWidget):
    """Shows animated GIFs - dancing during recording, celebration when done."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(250)
        self.setStyleSheet("background-color: #0b1322; border-radius: 8px; border: 1px solid #1d355d;")
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Create label for GIF
        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.gif_label)
        
        # Load dancing GIF (during recording)
        self.dancing_movie = QMovie("dancing_character.gif")
        
        # Load finished GIF (when recording ends)
        self.finished_movie = QMovie("finished_celebration.gif")
        
        # Initially show dancing GIF but hidden
        self.gif_label.setMovie(self.dancing_movie)
        self.dancing_movie.start()
        self.gif_label.setVisible(False)
        
        # Add status text
        self.status_label = QLabel("Ready to recognize music...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #8ea8d6; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.status_label)
    
    def start_dancing(self):
        """Show dancing GIF during recording."""
        self.gif_label.setVisible(True)
        self.gif_label.setMovie(self.dancing_movie)
        self.dancing_movie.start()
        self.finished_movie.stop()
        self.status_label.setText("Recording... Dance party in progress! ")
        self.status_label.setStyleSheet("color: #00e5ff; font-size: 14px; font-weight: bold;")
    
    def show_finished(self):
        """Show finished/celebration GIF when recording ends."""
        self.gif_label.setVisible(True)
        self.gif_label.setMovie(self.finished_movie)
        self.finished_movie.start()
        self.dancing_movie.stop()
        self.status_label.setText("Done! Great performance!")
        self.status_label.setStyleSheet("color: #ffb74d; font-size: 14px; font-weight: bold;")
    
    def reset(self):
        """Reset to ready state."""
        self.gif_label.setVisible(False)
        self.dancing_movie.stop()
        self.finished_movie.stop()
        self.status_label.setText("🎵 Ready to recognize music...")
        self.status_label.setStyleSheet("color: #8ea8d6; font-size: 14px; font-weight: bold;")
# gui/music_gui.py
import json
import csv
import webbrowser
from datetime import datetime
from urllib.parse import quote_plus

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QRadioButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QButtonGroup, QFrame, QMessageBox,
    QAbstractItemView, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QCursor, QMovie

from gui.worker import RecognitionWorker, HUMMING_AVAILABLE


class MusicGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAY 14 of MUSIC - Professional Studio")
        self.resize(1100, 760)
        self.setMinimumSize(980, 680)
        self.setStyleSheet("background-color: #070b14; color: #e8f2ff;")

        self.worker = None
        self.match_history = []
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # --- HEADER ---
        title = QLabel("🎵 Music Retrieval Engine")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: #00e5ff;")
        main_layout.addWidget(title)

        subtitle = QLabel("Professional Audio Recognition • Original / Cover / Humming")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: #8ea8d6; margin-bottom: 10px;")
        main_layout.addWidget(subtitle)

        # --- CONTROLS CARD ---
        controls_frame = QFrame()
        controls_frame.setStyleSheet("background-color: #10192a; border-radius: 8px;")
        controls_layout = QHBoxLayout(controls_frame)
        controls_layout.setContentsMargins(20, 20, 20, 20)

        # Duration
        dur_layout = QVBoxLayout()
        dur_label = QLabel("Duration (sec)")
        dur_label.setStyleSheet("color: #cfe2ff; font-weight: bold;")
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["5", "10", "15", "20", "25", "30"])
        self.duration_combo.setCurrentText("10")
        self.duration_combo.setStyleSheet(
            "background-color: #0d1422; border: 1px solid #1d355d; padding: 5px;"
        )
        dur_layout.addWidget(dur_label)
        dur_layout.addWidget(self.duration_combo)
        controls_layout.addLayout(dur_layout)

        # Mode Selection
        mode_layout = QVBoxLayout()
        mode_label = QLabel("Processing Mode")
        mode_label.setStyleSheet("color: #cfe2ff; font-weight: bold;")

        self.mode_group = QButtonGroup(self)
        radio_layout = QHBoxLayout()

        self.rb_orig = QRadioButton("Original")
        self.rb_cover = QRadioButton("Cover")
        self.rb_hum = QRadioButton("Humming")
        self.rb_orig.setChecked(True)

        for rb in (self.rb_orig, self.rb_cover, self.rb_hum):
            rb.setStyleSheet(
                "QRadioButton { color: #8ea8d6; } "
                "QRadioButton::indicator:checked { background-color: #00e5ff; border: 2px solid white; }"
            )
            self.mode_group.addButton(rb)
            radio_layout.addWidget(rb)

        mode_layout.addWidget(mode_label)
        mode_layout.addLayout(radio_layout)
        controls_layout.addLayout(mode_layout)

        controls_layout.addStretch()

        # Action Buttons
        self.btn_recognize = QPushButton("RECOGNIZE AUDIO")
        self.btn_recognize.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_recognize.setStyleSheet("""
            QPushButton { background-color: #00e5ff; color: #001018;
                          font-weight: bold; font-size: 14px;
                          padding: 12px 24px; border-radius: 4px; }
            QPushButton:hover { background-color: #00b8d4; }
            QPushButton:disabled { background-color: #1d355d; color: #8ea8d6; }
        """)
        self.btn_recognize.clicked.connect(self.start_recognition)
        controls_layout.addWidget(self.btn_recognize)

        self.btn_export = QPushButton("EXPORT RESULTS")
        self.btn_export.setCursor(QCursor(Qt.PointingHandCursor))
        self.btn_export.setStyleSheet("""
            QPushButton { background-color: #10192a; color: #00e5ff;
                          font-weight: bold; font-size: 12px;
                          padding: 12px 20px; border-radius: 4px;
                          border: 1px solid #00e5ff; }
            QPushButton:hover { background-color: #00e5ff; color: #001018; }
            QPushButton:disabled { background-color: #1d355d; color: #8ea8d6;
                                   border: 1px solid #1d355d; }
        """)
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        controls_layout.addWidget(self.btn_export)

        main_layout.addWidget(controls_frame)

        # --- SINGLE GIF AREA ---
        self.gif_label = QLabel()
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setFixedHeight(250)
        self.gif_label.setStyleSheet(
            "background-color: #0b1322; border-radius: 8px; border: 1px solid #1d355d;"
        )

        self.dancing_movie = QMovie("dancing_character.gif")
        self.finished_movie = QMovie("finished_celebration.gif")

        self.gif_label.setMovie(self.dancing_movie)
        self.dancing_movie.start()
        self.gif_label.setVisible(False)

        self.gif_status = QLabel("🎵 Ready to recognize music...")
        self.gif_status.setAlignment(Qt.AlignCenter)
        self.gif_status.setStyleSheet("color: #8ea8d6; font-size: 14px; font-weight: bold;")

        gif_container = QWidget()
        gif_layout = QVBoxLayout(gif_container)
        gif_layout.setContentsMargins(0, 0, 0, 0)
        gif_layout.addWidget(self.gif_label)
        gif_layout.addWidget(self.gif_status)
        main_layout.addWidget(gif_container)

        # --- RESULTS CARD ---
        results_frame = QFrame()
        results_frame.setStyleSheet("background-color: #10192a; border-radius: 8px;")
        results_layout = QVBoxLayout(results_frame)

        self.result_title = QLabel("Ready to listen.")
        self.result_title.setFont(QFont("Segoe UI", 13))
        self.result_title.setStyleSheet("color: #e8f2ff;")
        self.result_title.setWordWrap(True)
        results_layout.addWidget(self.result_title)

        self.yt_link = QLabel("")
        self.yt_link.setStyleSheet("color: #8ea8d6; text-decoration: underline;")
        self.yt_link.setCursor(QCursor(Qt.PointingHandCursor))
        self.yt_link.mousePressEvent = self.open_youtube
        results_layout.addWidget(self.yt_link)

        main_layout.addWidget(results_frame)

        # --- CANDIDATE TABLE ---
        self.table_label = QLabel(
            "Search Results Leaderboard (Double-click any row to open on YouTube)"
        )
        self.table_label.setFont(QFont("Segoe UI", 11))
        self.table_label.setStyleSheet("color: #8ea8d6;")
        main_layout.addWidget(self.table_label)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "Candidate Song", "Match Score / Distance"]
        )
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.cellDoubleClicked.connect(self.recommendation_clicked)
        self.table.setCursor(QCursor(Qt.PointingHandCursor))
        self.table.setStyleSheet("""
            QTableWidget { background-color: #0d1422; border: 1px solid #1d355d;
                           color: #cfe2ff; gridline-color: #1d355d; }
            QHeaderView::section { background-color: #10192a; color: #00e5ff;
                                   font-weight: bold; border: none; padding: 5px; }
            QTableWidget::item:selected { background-color: #1d355d; color: #00e5ff; }
            QTableWidget::item:hover { background-color: #162a4a; }
        """)
        main_layout.addWidget(self.table)

        # --- MATCH HISTORY ---
        history_frame = QFrame()
        history_frame.setStyleSheet("background-color: #10192a; border-radius: 8px;")
        history_layout = QVBoxLayout(history_frame)
        history_label = QLabel("📋 Match History (Last 10)")
        history_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        history_label.setStyleSheet("color: #00e5ff;")
        history_layout.addWidget(history_label)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(
            ["Time", "Mode", "Result", "Confidence"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setCursor(QCursor(Qt.PointingHandCursor))
        self.history_table.setStyleSheet("""
            QTableWidget { background-color: #0d1422; border: 1px solid #1d355d;
                           color: #cfe2ff; gridline-color: #1d355d; }
            QHeaderView::section { background-color: #10192a; color: #00e5ff;
                                   font-weight: bold; border: none; padding: 5px; }
            QTableWidget::item:selected { background-color: #1d355d; color: #00e5ff; }
        """)
        history_layout.addWidget(self.history_table)
        main_layout.addWidget(history_frame)

        # --- STATUS BAR ---
        self.status_bar = QLabel("System Idle.")
        self.status_bar.setStyleSheet("color: #8ea8d6;")
        main_layout.addWidget(self.status_bar)

    # ---------- LOGIC ----------

    def start_recognition(self):
        duration = int(self.duration_combo.currentText())
        mode = "original"
        if self.rb_cover.isChecked():
            mode = "cover"
        elif self.rb_hum.isChecked():
            mode = "humming"

        self.current_mode = mode

        if mode == "humming" and not HUMMING_AVAILABLE:
            QMessageBox.critical(self, "Error", "Humming module not found.")
            return

        self.btn_recognize.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.result_title.setText("Listening...")
        self.result_title.setStyleSheet("color: #8ea8d6;")
        self.yt_link.setText("")
        self.table.setRowCount(0)

        # Show dancing GIF during recording
        self.gif_label.setVisible(True)
        self.gif_label.setMovie(self.dancing_movie)
        self.dancing_movie.start()
        self.finished_movie.stop()
        self.gif_status.setText("🎵 Recording... Dance party in progress! 🎵")
        self.gif_status.setStyleSheet(
            "color: #00e5ff; font-size: 14px; font-weight: bold;"
        )

        self.worker = RecognitionWorker(duration, mode)
        self.worker.status_update.connect(self.status_bar.setText)
        # We could use wave_active to drive something else, but we keep GIF only
        self.worker.finished_success.connect(self.on_success)
        self.worker.finished_fail.connect(self.on_fail)
        self.worker.start()

    def on_success(self, song_name, list_data, error):
        self.btn_recognize.setEnabled(True)
        self.btn_export.setEnabled(True)

        # Switch to finished GIF
        self.gif_label.setMovie(self.finished_movie)
        self.finished_movie.start()
        self.dancing_movie.stop()
        self.gif_status.setText("🎉 Done! Great performance! 🎉")
        self.gif_status.setStyleSheet(
            "color: #ffb74d; font-size: 14px; font-weight: bold;"
        )

        # Same display logic as music_gui_pro.py
        if song_name == "Uncertain / Multiple Candidates":
            self.result_title.setText("Top Candidates Detected (Uncertain Match)")
            self.result_title.setStyleSheet("color: #ffb74d;")
            self.yt_link.setText("")
            self.status_bar.setText(f"Status: {error}")
        else:
            self.result_title.setText(f"Top Match Candidate: {song_name}")
            self.result_title.setStyleSheet("color: #e8f2ff;")
            self.current_yt_url = (
                f"https://www.youtube.com/results?search_query={quote_plus(song_name)}"
            )
            self.yt_link.setText(
                f"🔍 Click to search top candidate on YouTube: {song_name}"
            )
            self.status_bar.setText(
                "Recognition Complete." if not error else f"Warning: {error}"
            )

        # Populate table
        self.table.setRowCount(len(list_data))
        for row, (name, score) in enumerate(list_data):
            self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            item_name = QTableWidgetItem(name)
            item_name.setForeground(QColor("#cfe2ff"))
            self.table.setItem(row, 1, item_name)
            self.table.setItem(row, 2, QTableWidgetItem(f"{score:.2f}"))

        self._add_to_history(self.current_mode, song_name, list_data)
        self.current_result = {
            "song": song_name,
            "leaderboard": list_data,
            "mode": self.current_mode,
            "error": error,
        }

    def on_fail(self, reason):
        self.btn_recognize.setEnabled(True)
        self.btn_export.setEnabled(False)

        # Even on fail, show finished GIF so user sees "done"
        self.gif_label.setMovie(self.finished_movie)
        self.finished_movie.start()
        self.dancing_movie.stop()
        self.gif_status.setText("🎉 Done! Great performance! 🎉")
        self.gif_status.setStyleSheet(
            "color: #ffb74d; font-size: 14px; font-weight: bold;"
        )

        self.result_title.setText("Not Recognized.")
        self.result_title.setStyleSheet("color: #e57373;")
        self.status_bar.setText(reason)
        self.yt_link.setText("")
        self._add_to_history("failed", "No Match", [], reason)

    def _add_to_history(self, mode, result, leaderboard, error=""):
        timestamp = datetime.now().strftime("%H:%M:%S")
        confidence = "N/A"
        if leaderboard:
            top_score = leaderboard[0][1]
            max_score = max(c[1] for c in leaderboard)
            ratio = top_score / max_score if max_score > 0 else 0
            confidence = f"{ratio:.0%}"

        self.match_history.insert(
            0,
            {
                "time": timestamp,
                "mode": mode,
                "result": result,
                "confidence": confidence,
                "error": error,
            },
        )
        self.match_history = self.match_history[:10]
        self.history_table.setRowCount(len(self.match_history))
        for i, entry in enumerate(self.match_history):
            self.history_table.setItem(i, 0, QTableWidgetItem(entry["time"]))
            self.history_table.setItem(i, 1, QTableWidgetItem(entry["mode"]))
            self.history_table.setItem(i, 2, QTableWidgetItem(entry["result"]))
            self.history_table.setItem(i, 3, QTableWidgetItem(entry["confidence"]))

    def export_results(self):
        if not hasattr(self, "current_result"):
            QMessageBox.warning(self, "No Results", "No recognition results to export.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".json"):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.current_result, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Rank", "Song Name", "Score"])
                    for i, (name, score) in enumerate(
                        self.current_result["leaderboard"], 1
                    ):
                        writer.writerow([i, name, f"{score:.2f}"])
            QMessageBox.information(
                self, "Export Successful", f"Results exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Error: {str(e)}")

    def open_youtube(self, event=None):
        if hasattr(self, "current_yt_url") and "Uncertain" not in self.result_title.text():
            webbrowser.open_new_tab(self.current_yt_url)

    def recommendation_clicked(self, row, column):
        song_name = self.table.item(row, 1).text()
        url = f"https://www.youtube.com/results?search_query={quote_plus(song_name)}"
        webbrowser.open_new_tab(url)
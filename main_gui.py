# main_gui.py
import sys
from PyQt5.QtWidgets import QApplication
from gui.music_gui import MusicGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Optional: cleaner cross-platform look
    window = MusicGUI()
    window.show()
    sys.exit(app.exec_())
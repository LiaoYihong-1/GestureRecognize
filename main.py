from gui import CameraApp
from PyQt5.QtWidgets import QApplication
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    sys.exit(app.exec_())
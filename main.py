import dxcam
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QMainWindow
from win32gui import FindWindow, GetClientRect, ClientToScreen
import cv2
import sys
import mouse
import tensorflow as tf
from tensorflow import keras

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My App")
        self.unitUI()

    def unitUI(self):
        self.widget = QtWidgets.QLabel(self)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_image)
        timer.start(1)

    def update_image(self):
        res_img = cv2.resize(camera.get_latest_frame(), (250, 125), cv2.INTER_AREA)
        # Image.fromarray(res_img).show()

        x, y = mouse.get_position()
        print(f"Курсор находится в точке: ({x}, {y})")

        h, w, ch = res_img.shape
        bytes_per_line = ch * w

        qimg = QImage(
            res_img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimg)
        self.widget.setScaledContents(True)
        self.widget.setPixmap(pixmap)

        self.setCentralWidget(self.widget)

window_handle = FindWindow(None, "osu!")
l, t, r, b = GetClientRect(window_handle)
cl, ct = ClientToScreen(window_handle, (l, t))

size = (r - l, b - t)
region = (cl, ct, cl + size[0], ct + size[1])

camera = dxcam.create()
camera.start(region=region)
# for x in range(1):
#     res_img = cv2.resize(camera.get_latest_frame(), (320, 180), cv2.INTER_AREA)
#     Image.fromarray(res_img).show()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()

camera.stop()
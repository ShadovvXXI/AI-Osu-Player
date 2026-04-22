from PyQt6.QtWidgets import QApplication
import sys
import logging

from player import Player, RECORD, TRAIN, PLAY

logging.basicConfig(level=logging.INFO)

# размеры изображения для нейросети
WIDTH = 150
HEIGHT = 60

app = QApplication(sys.argv)

window = Player(["Gira Gira"], model_name="0.92", img_size=(WIDTH, HEIGHT), mode=TRAIN)
window.show()

app.exec()
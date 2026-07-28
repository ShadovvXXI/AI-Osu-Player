from PyQt6.QtWidgets import QApplication
import sys
import logging

from player import Player, RECORD, TRAIN, PLAY

logging.basicConfig(level=logging.INFO)

# размеры изображения для нейросети
WIDTH = 120
HEIGHT = 60

app = QApplication(sys.argv)

window = Player(["Gira Gira", "Rory", "Mirror", "Daidai Genome", "Jama", "Ordinary", "RuLe",
                 "Unravel", "Light", "Shooting Star"],
                model_name="0.96", img_size=(WIDTH, HEIGHT), mode=PLAY)
window.show()

app.exec()
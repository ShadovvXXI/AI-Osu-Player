import numpy as np
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow
from win32gui import FindWindow, GetClientRect, ClientToScreen
import dxcam
import time as tm
import pygetwindow as gw
from collections import deque
import cv2
import pickle
import logging
import mouse
import os

import torch
from torch.utils.data import DataLoader

from song import Song
from nn import (OsuImageDataset, OsuNeuralNetwork, prepare_data_for_prediction)
from stats import calculate_current_mean_std, calculate_stats_from_record, combine_stats
from utils import window_pos_to_train_pos, pred_pos_to_window_pos, draw_image_with_circle

RECORD = 0
TRAIN = 1
PLAY = 2


class Player(QMainWindow):
    def __init__(self, song_names, model_name, img_size, mode):
        super().__init__()

        # обработчик окна и его координаты на экране
        window_handle = FindWindow(None, "osu!")
        l, t, r, b = GetClientRect(window_handle)
        cl, ct = ClientToScreen(window_handle, (l, t))

        size = (r - l, b - t)
        region = (cl, ct, cl + size[0], ct + size[1])

        # по наблюдениям высота поля всегда 80% от общей высоты окна
        playfield_h = 0.8 * size[1]
        # соотношение поля всегда 3/4
        playfield_w = playfield_h * 4 / 3
        # вычисляем scale по длине с наименьшим коэфициентом изменяемости
        self.scale = playfield_h / 384

        # расстояние слева и справа всегда одинаковое
        offset_x = (size[0] - playfield_w) / 2
        # по наблюдениям смещение свеху всегда 11,6% от общей высоты, снизу - 8,3%
        offset_y = size[1] * 0.116

        self.offset = (offset_x, offset_y)

        # создаем область для записи и начинаем ее
        self.camera = dxcam.create(region=region, output_color="GRAY")
        self.camera.start()
        self.size = (self.camera.region[2] - self.camera.region[0], self.camera.region[3] - self.camera.region[1])

        self.mean, self.std = calculate_current_mean_std()

        self.mode = mode
        self.model_name = model_name
        self.model = OsuNeuralNetwork()
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model.to(self.device)
        if os.path.exists("./models/"+self.model_name+".pth"):
            self.model.load_state_dict(torch.load("./models/"+self.model_name+".pth"))
        self.loss_func = torch.nn.SmoothL1Loss()
        self.epochs = 10
        lr = 1e-3
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.player_deque = deque(maxlen=5)

        self.widget = QtWidgets.QLabel(self)
        self.setWindowTitle("My App")
        self.img_size = img_size

        self.songs = {}
        for name in song_names:
            self.songs[name] = {"file": self.load_song(name)}

        self.records = {}
        for name in song_names:
            self.load_from_file(name)

        self.start_timer = None
        self.starting = False
        self.recorded_images = dict()
        self.recorded_song_name = ""
        self.training_state = False
        self.training_time = 0
        self.skip_time = 280
        self.mode_manager()

    def __del__(self):
        # окончание записи экрана
        self.camera.stop()

    def mode_manager(self):
        if self.mode == RECORD:
            timer = QtCore.QTimer(self)
            timer.timeout.connect(self.millisecond_record_tick)
            timer.start(1)
        elif self.mode == TRAIN:
            if not self.mean and not self.std:
                stats = []
                for record in self.records:
                    stats.append(calculate_stats_from_record(self.records[record]))
                self.mean, self.std = combine_stats(stats, save_to_file=True)
            self.training_pipeline()
        elif self.mode == PLAY:
            self.mean, self.std = calculate_current_mean_std()
            timer = QtCore.QTimer(self)
            timer.timeout.connect(self.millisecond_playing_tick)
            timer.start(1)

    def starting_skip(self):
        self.start_timer = tm.perf_counter_ns()
        self.training_state = True
        self.recorded_song_name = [s for s in gw.getAllTitles() if "osu!" in s][0]
        self.starting = False

    def millisecond_record_tick(self):
        if "osu!" not in gw.getAllTitles() and any("osu!" in s for s in gw.getAllTitles()):
            if not self.training_state and not self.starting:
                self.starting = True
                timer = QtCore.QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(self.starting_skip)
                timer.start(self.skip_time)

        image = self.update_image()
        if self.training_state:
            elapsed_ms = (tm.perf_counter_ns() - self.start_timer) // 1_000_000
            self.recorded_images[elapsed_ms] = image

        if "osu!" in gw.getAllTitles() and self.training_state:
            self.sync_image_to_pos(save_to_file=True)

            self.recorded_images = dict()
            self.training_state = False
            self.training_time += 1

    def training_pipeline(self):
        for record in self.records:
            dataset = OsuImageDataset(self.records[record], self.mean, self.std)
            # TODO : рассмотреть разные batch_size
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            for t in range(self.epochs):
                print(f"Epoch: {t + 1}\n-------------------------------")
                self.train_model(dataloader, self.loss_func, self.optimizer, self.device)
                self.test_model(dataloader, self.loss_func, self.device)

            torch.save(self.model.state_dict(), "./models/" + self.model_name + ".pth")

    def millisecond_playing_tick(self):
        image = self.update_image()
        if "osu!" not in gw.getAllTitles() and any("osu!" in s for s in gw.getAllTitles()):
            if len(self.player_deque) == 5:
                self.player_deque.append(image)
                with torch.no_grad():
                    prepared_data = prepare_data_for_prediction(np.array(self.player_deque), self.mean, self.std).to(self.device)
                    pred = self.model(prepared_data.unsqueeze(0))
                    pos = pred_pos_to_window_pos(self.size, (pred[0, 0].item(), pred[0, 1].item()), self.img_size[0], self.img_size[1])
                    mouse.move(*pos)
            else:
                self.player_deque.append(image)

    def update_image(self):
        res_img = cv2.resize(self.camera.get_latest_frame(), self.img_size, interpolation=cv2.INTER_AREA)

        # преобразуем в формат подходящий для Qt
        h, w = res_img.shape
        bytes_per_line = w

        qimg = QImage(
            res_img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_Grayscale8
        )

        pixmap = QPixmap.fromImage(qimg)
        self.widget.setScaledContents(True)
        self.widget.setPixmap(pixmap)

        self.setCentralWidget(self.widget)

        return res_img

    def load_song(self, name):
        new_song = Song(name)
        new_song.parse_map_file(name)
        new_song.build_beatmap()
        if not new_song.load_from_file():
            new_song.sync_timings_to_pos(self.camera.region, self.scale, self.offset, save_to_file=True)
        return new_song

    def sync_image_to_pos(self, save_to_file):
        for song in self.songs:
            if song.lower() in self.recorded_song_name.lower():
                pos = self.songs[song]["file"].hit_timings_to_pos
                max_pos = max(pos)

                for moment in sorted(self.recorded_images):
                    timing = (moment - self.songs[song]["file"].lead_in +
                              (self.skip_time*1.3 if self.songs[song]["file"].lead_in else 0))
                    if timing > max_pos:
                        break

                    if timing in pos:
                        current_pos = pos[timing]
                    elif pos[timing-1]:
                        current_pos = pos[timing-1]
                    else:
                        continue

                    self.records[song][timing] = {
                        "pos": window_pos_to_train_pos(self.size, current_pos, self.img_size[0], self.img_size[1]),
                        "image": self.recorded_images[moment]
                    }

                    # debug func
                    # if moment > 3000:
                    #     draw_image_with_circle(self.songs[song][timing]["image"], self.songs[song][timing]["pos"])

                if save_to_file: self.save_to_file(song)
                self.recorded_song_name = song
                break

    def save_to_file(self, song_name):
        if self.records[song_name]:
            with open("records\\"+song_name+".pkl", "wb") as f:
                pickle.dump(self.records[song_name], f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_file(self, song_name):
        try:
            with open("records\\"+song_name+".pkl", "rb") as f:
                self.records[song_name] = pickle.load(f)
                logging.info("Time_to_img_and_pos file loaded")
                return True
        except Exception as e:
            logging.info("Record file for " + song_name + " song corrupted or not find: " + str(e))
            self.records[song_name] = {}
            return False

    def train_model(self, dataloader, loss_fn, optimizer, device):
        size = len(dataloader.dataset)

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = self.model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_model(self, dataloader, loss_fn, device):
        self.model.eval()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss = 0
        mean_euclidean_dist = torch.Tensor([0]).to(device)

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)

                pred = self.model(X)
                test_loss += loss_fn(pred, y).item()
                mean_euclidean_dist += torch.norm(pred - y, dim=1)

        test_loss /= num_batches
        mean_euclidean_dist /= size
        print(f"test error: \n mean_euclidean_dist: {(mean_euclidean_dist.item()):>0.1f}, avg loss: {test_loss:>8f} \n")
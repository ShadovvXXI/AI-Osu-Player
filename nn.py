import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset

def prepare_data_for_prediction(data, mean, std, transform=transforms.Normalize):
    data = torch.from_numpy(data).float() / 255.0
    transformation = transform([mean] * 5, [std] * 5)
    return transformation(data)

class OsuImageDataset(Dataset):
    def __init__(self, songs, mean, std, transform=transforms.Normalize, target_transform=None):
        self.len = 0
        self.songs = [[] for _ in range(len(songs))]
        for idx, song in enumerate(songs):
            for moment in sorted(songs[song]):
                if len(self.songs[idx])==0 or (len(self.songs[idx])>0 and self.songs[idx][-1]["pos"] != songs[song][moment]["pos"]):
                    self.songs[idx].append(songs[song][moment])
            self.len += len(self.songs[idx])-4

        self.transform = transform([mean]*5, [std]*5)
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        song_idx = 0
        for song in range(len(self.songs)):
            song_len_sub4 = len(self.songs[song])-1 - 4
            if idx < song_len_sub4:
                song_idx = song
                break
            else:
                idx -= song_len_sub4

        image = torch.from_numpy(np.stack([self.songs[song_idx][x]["image"] for x in range(idx, idx + 5)], axis=0)).float() / 255.0
        label = torch.tensor(self.songs[song_idx][idx+4]["pos"])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class OsuNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool_part = nn.Sequential(
            nn.Conv2d(5, 64, 5, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1),
            nn.ReLU(),
        )
        self.linear_part = nn.Sequential(
            nn.Linear(12288, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.conv_pool_part(x)
        x = nn.Flatten()(x)
        x = self.linear_part(x)
        return x
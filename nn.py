import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset

def prepare_data_for_prediction(data, mean, std, transform=transforms.Normalize):
    data = torch.from_numpy(data).float() / 255.0
    transformation = transform([mean] * 5, [std] * 5)
    return transformation(data)

def target_transform_func(label, img_size):
    x = label[0] / img_size[0] * 2 - 1
    y = label[1] / img_size[1] * 2 - 1
    return torch.tensor([x, y])

class OsuImageDataset(Dataset):
    def __init__(self, songs, mean, std, img_size, transform=transforms.Normalize, target_transform=target_transform_func):
        self.len = 0
        self.songs = [[] for _ in range(len(songs))]
        for idx, song in enumerate(songs):
            for moment in sorted(songs[song]):
                if len(self.songs[idx])==0 or (len(self.songs[idx])>0 and self.songs[idx][-1]["pos"] != songs[song][moment]["pos"]):
                    self.songs[idx].append(songs[song][moment])
            self.len += len(self.songs[idx])-4

        self.transform = transform([mean]*5, [std]*5)
        self.img_size = img_size
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
            label = self.target_transform(label, self.img_size)
        return image, label

class OsuNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool_part = nn.Sequential(
            nn.Conv2d(1, 64, 5, 3), #39x19
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 5, 2), #18x8
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1),  # 16x6
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1),  # 14x4
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 1, 1),  # 14x4
            nn.LeakyReLU(),
            # nn.AdaptiveAvgPool2d(1)
        )
        self.recurent_part = nn.Sequential(
            nn.GRU(3584, 64, 2, batch_first=True)
        )
        self.linear_part = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out = x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3])
        out = self.conv_pool_part(out)
        out = nn.Flatten(1, 3)(out)
        out = out.view(x.shape[0], x.shape[1], out.shape[1])
        out, hn = self.recurent_part(out)
        out = self.linear_part(hn[-1])
        return out
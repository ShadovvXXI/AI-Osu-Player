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
    def __init__(self, song, mean, std, transform=transforms.Normalize, target_transform=None):
        self.song = []
        for moment in sorted(song):
            self.song.append(song[moment])

        self.transform = transform([mean]*5, [std]*5)
        self.target_transform = target_transform

    def __len__(self):
        return len(self.song)-4

    def __getitem__(self, idx):
        image = torch.from_numpy(np.stack([self.song[x]["image"] for x in range(idx, idx + 5)], axis=0)).float() / 255.0
        label = torch.tensor(self.song[idx+4]["pos"]) # регрессия последнего кадра, возможно нужно сделать предсказание последующего
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class OsuNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool_part = nn.Sequential(
            nn.Conv2d(5, 32, 3, 2),
            nn.Conv2d(32, 64, 3, 2),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2)
        )
        self.linear_part = nn.Sequential(
            nn.Linear(6720, 100),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        x = self.conv_pool_part(x)
        x = nn.Flatten()(x)
        x = self.linear_part(x)
        return x

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = OsuNeuralNetwork().to(device)
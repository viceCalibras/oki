from torch.utils.data import Dataset, DataLoader
import pathlib
from typing import Union

# TODO(viceCalibras) Split the test & validation datasets! Also in notebooks!
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class SimulationDataset(Dataset):
    def __init__(self, dataset_path: Union[pathlib.Path, str], features: list, labels: list):
        """Pytorch Dataset object for Deform 2D simulation data.
        """
        self.df = pd.read_csv(dataset_path)
        self.features = self.df[features]
        self.labels = self.df[labels]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.Tensor(self.features.iloc[idx].values)
        label = torch.Tensor(self.labels.iloc[idx].values)
        return feature, label
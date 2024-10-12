from torch.utils.data import Dataset, DataLoader

# TODO(viceCalibras) Split the test & validation datasets! Also in notebooks!
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, _features, _labels):
        self.features = _features
        self.labels = _labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.Tensor(self.features.iloc[idx].values)
        label = torch.Tensor(self.labels.iloc[idx].values)
        return feature, label


csv_file_path = "data/thesis_data_cleaned.csv"
df = pd.read_csv(csv_file_path)
# features = df[["Ton", "Toff", "Id", "SV"]]
features = df[["Depth Of Cut", "Feed Rate", "Lenght Of Cut"]]
# labels = df[["KW", "MRR", "WLT", "Ra"]]
labels = df[["Load X", "Load Y"]]

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.1, random_state=42
)

batch_size = 4  # Set your desired batch size

train_dataset = CustomDataset(train_features, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

test_dataset = CustomDataset(test_features, test_labels)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from data_loader import train_dataloader, test_dataloader
from model import model

model_path = "./trained_models/new_model_state_dict"

model.load_state_dict(torch.load(model_path))

model.eval()  # Set the model to evaluation mode

for batch_features, batch_labels in test_dataloader:
    outputs = model(batch_features)
    predicted_labels = outputs.cpu().detach().numpy()
    # predicted_labels = outputs.cpu().numpy()

    for i in range(len(batch_features)):
        features = batch_features[i]
        labels = batch_labels[i]
        predicted = predicted_labels[i]

        print("Features:")
        print(features)

        print("Predicted Labels:")
        print(predicted)

        print("Actual Labels:")
        print(labels)

        print("------------------")

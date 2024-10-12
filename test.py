import torch
from data_loader import test_dataloader
from model import model

# Set the device & clean the memory
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
torch.cuda.empty_cache()

model_path = "./trained_models/new_model_state_dict"

model.load_state_dict(torch.load(model_path))

model.eval()  # Set the model to evaluation mode

for batch_features, batch_labels in test_dataloader:
    outputs = model(batch_features)
    if device == "cuda:0":
        predicted_labels = outputs.cpu().numpy()
    else:
        predicted_labels = outputs.cpu().detach().numpy()

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

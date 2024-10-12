from model import model
from train import train, save_model_and_loss
import torch
import torch.optim as optim
from data_loader import train_dataloader, test_dataloader

# Set the device & clean the memory
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
torch.cuda.empty_cache()

# Set the loss function and optimization algorithm.
loss_f = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.to(device)
trained_model, train_loss, valid_loss = train(
    model, device, 10, loss_f, optimizer, train_dataloader, test_dataloader
)

save_model_and_loss("./trained_models", model, "new_model", train_loss, valid_loss)

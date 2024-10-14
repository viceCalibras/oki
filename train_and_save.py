import gc
import time
from typing import List, Tuple, Union
import torch
from torch.utils.data import random_split, DataLoader
import pathlib
import tqdm
import types
from model import MLPReg
import torch.optim as optim
from dataset import SimulationDataset


def save_model_and_loss(
    output_dir: Union[pathlib.Path, str],
    model: torch.nn.Module,
    model_name: str,
    train_loss: List[float],
    valid_loss: List[float],
) -> None:
    """Saves the trained model & its loss function values in each training step.

    Args:
        output_dir: Absolute path to the output folder.
        model: Trained model.
        model_name: Name of the trained model.
        train_loss: Loss data generated during training.
        valid_loss: Loss data generated during validation.
    """
    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists:
        pathlib.mkdir(output_dir)

    torch.save(model, output_dir / (model_name + ".pt"))
    torch.save(model.state_dict(), output_dir / (model_name + "_state_dict"))

    with open(output_dir / (model_name + "_training_loss.txt"), "w") as f:
        for element in train_loss:
            f.write(str(element) + "\n")
    with open(output_dir / (model_name + "_validation_loss.txt"), "w") as f:
        for element in valid_loss:
            f.write(str(element) + "\n")
    print("Model, state dictionary and loss values saved at: ", str(output_dir))


def train(
    model: torch.nn.Module,
    device: torch.device,
    epochs: int,
    loss_f: types.ModuleType,
    optimizer: types.ModuleType,
    train_loader: torch.utils.data.dataloader.DataLoader,
    valid_loader: torch.utils.data.dataloader.DataLoader,
    feature_weights: Union[torch.tensor, None] = None
) -> Tuple[torch.nn.Module, List[float], List[float]]:
    """Performs training of the input model.

    Args:
        model: Model that is to be trained.
        device: PyTorch device (gpu or cpu) aka. hardware that will be used for computation.
        epochs: Number of training epochs.
        loss_f: Loss function.
        optimizer: Learning grade optimizer.
        train_loader: PyTorch DataLoader object, iterator through training dataset.
        valid_loader: PyTorch DataLoader object, iterator through validation dataset.
        feature_weights: A tensor of weights to be multiplied by the features before passing them   to the model.

    Returns:
       Trained model and loss function values.
    """
    gc.collect()  # Garbage colection.
    train_losses_all = []
    valid_losses_all = []
    time_start = time.time()
    print("Training the model!")

    for epoch in tqdm.trange(epochs):
        # Training loop.
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            # Clear the gradients.
            optimizer.zero_grad()
            # Forward pass.
            if feature_weights is not None:
                feature_weights = feature_weights.to(device)
                X = feature_weights * X
            y_hat = model(X)
            y_hat = torch.squeeze(y_hat)  # To match the dimensions of y.
            # Compute the loss.
            loss = loss_f(y_hat, y)
            # Backpropagate the loss & compute gradients.
            loss.backward()
            # Update weights.
            optimizer.step()

            train_loss += loss.item()

        train_loss_in_epoch = train_loss / len(train_loader)

        # Validation loop.
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():  # Turn off the gradients for validation.
            for X, y in valid_loader:
                X = X.to(device)
                y = y.to(device)
                # Forward pass.
                y_hat = model(X)
                y_hat = torch.squeeze(y_hat)  # To match the dimensions of y.
                # Compute the loss.
                loss = loss_f(y_hat, y)

                valid_loss += loss.item()

        valid_loss_in_epoch = valid_loss / len(valid_loader)

        print("Epoch", epoch + 1, "complete!"),
        print("\tTraining Loss: ", round(train_loss_in_epoch, 4))
        print("\tValidation Loss: ", round(valid_loss_in_epoch, 4))
        train_losses_all.append(train_loss_in_epoch)
        valid_losses_all.append(valid_loss_in_epoch)

    time_end = time.time()
    print("Training finished!")
    print(f"Total training time: {int((time_end - time_start) / 60)} min.")

    return model, train_losses_all, valid_losses_all


if __name__ == "__main__":
    # Set the device & clean the memory
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    torch.cuda.empty_cache()

    # Load the data.
    dataset_path = "./data/dataset_train.csv"
    features = ["Temperature", "Depth Of Cut", "Feed Rate", "Lenght Of Cut"]
    labels = ["Load X", "Load Y"]

    # feature_weights = torch.tensor([0.8, 1.0, 1.0, 1.0])
    
    dataset = SimulationDataset(dataset_path, features, labels)

    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(42))

    batch_size = 8

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

    # Load the model.
    model = MLPReg()
    model.to(device)
    
    # Set the loss function and optimization algorithm.
    # loss_f = torch.nn.MSELoss(reduction='mean')
    loss_f = torch.nn.HuberLoss(delta=0.6)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    
    # Train and save the model.
    trained_model, train_loss, valid_loss = train(
        model, device, 60000, loss_f, optimizer, train_dataloader, validation_dataloader
    )

    save_model_and_loss("./trained_models", model, "lr_1_e_m5_b_8_e_60000_32_64_128_64_huber_6em1", train_loss, valid_loss)

import torch
import torch.optim as optim
from models.model import model
from data.dataLoader import *
from utils.logger import app_log


def train(num_epochs,learning_rate):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0

        for batch_features, batch_labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        app_log.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0

        if epoch % 200 == 0:
            model_path = f'training/checkpoints/model_{epoch}.pt'
            torch.save(model.state_dict(), model_path)
            app_log.info(f"Trained model saved at {model_path}")

        with torch.no_grad():
            for batch_features, batch_labels in test_dataloader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()

            avg_test_loss = test_loss / len(test_dataloader)
            app_log.info(f"Test Loss: {avg_test_loss:.4f}")

    for batch_features, batch_labels in test_dataloader:
        outputs = model(batch_features)
        predicted_labels = outputs.detach().numpy()

        for i in range(len(batch_features)):
            features = batch_features[i]
            labels = batch_labels[i]
            predicted = predicted_labels[i]

            app_log.info("Features:")
            app_log.info(features)

            app_log.info("Predicted Labels:")
            app_log.info(predicted)

            app_log.info("Actual Labels:")
            app_log.info(labels)

            app_log.info("------------------")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.resnet_model import get_resnet_model
from utils.dataset import AnimalDetectDataset

# Load dataset
train_dataset = AnimalDetectDataset("C:/Users/Batuhan/Desktop/codes/Animal_Detection/train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Get model and device
model, device = get_resnet_model()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")


# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=10)

# Save the model
torch.save(model.state_dict(), "animal_detect.pth")
print("Model saved successfully!")
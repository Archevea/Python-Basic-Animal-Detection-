import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from models.resnet_model import get_resnet_model
from utils.dataset import AnimalDetectDataset

# Load test dataset
test_dataset = AnimalDetectDataset("C:/Users/Batuhan/Desktop/codes/Animal_Detection/val")
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load model
model, device = get_resnet_model()
model.load_state_dict(torch.load("animal_detect.pth"))
model.eval()

# Evaluate
def evaluate_model(model, test_loader):
    predictions, true_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())

    acc = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {acc:.4f}")

evaluate_model(model, test_loader)
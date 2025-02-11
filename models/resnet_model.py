import torch
import torch.nn as nn
from torchvision import models


def get_resnet_model(num_classes=3):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Move to MPS (Apple GPU) if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device
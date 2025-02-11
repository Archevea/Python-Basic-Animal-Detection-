import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

# Define class labels
class_labels = {
    "cat": 0,
    "dog": 1,
    "wild": 2,
}

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


class AnimalDetectDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for animal_type, label in class_labels.items():
            folder_path = os.path.join(root_dir, animal_type)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
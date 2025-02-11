import torch
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_resnet_model

# Load model
model, device = get_resnet_model()
model.load_state_dict(torch.load("animal_detect.pth"))
model.eval()

# Define class names
class_names = ["cat", "dog", "wild"]

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    return class_names[predicted_class.item()]

# Example usage
image_path = "test.webp"  # Replace with actual image
print(f"Prediction: {predict_image(image_path)}")
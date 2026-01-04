import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from .model_arch import SimpleCNN

# CIFAR-10 classes
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class Predictor:
    def __init__(self, model_path='model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(num_classes=10).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Warning: Model file not found at {model_path}. Using random weights.")
        
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def predict(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        idx = predicted.item()
        return {
            "class_name": CLASSES[idx],
            "confidence": float(confidence.item())
        }

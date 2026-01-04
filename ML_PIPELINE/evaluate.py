import torch
from .model import SimpleCNN
from .data_loader import get_data_loaders
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path='./backend/model.pth', batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    _, _, test_loader = get_data_loaders(batch_size=batch_size)
    
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']))
    
if __name__ == "__main__":
    evaluate_model()

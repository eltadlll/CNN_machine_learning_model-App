import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from .preprocessing import get_train_transforms, get_test_transforms

def get_data_loaders(data_dir='./data', batch_size=64, val_split=0.1, num_workers=2):
    """
    Downloads CIFAR-10 and returns train, val, and test DataLoaders.
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load datasets
    train_dataset_full = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_train_transforms()
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_test_transforms()
    )

    # Split train into train/val
    val_size = int(len(train_dataset_full) * val_split)
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size])

    # Re-implementation for correct transform handling:
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    full_train_data = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
    
    # Generate split indices
    num_train = len(full_train_data)
    indices = list(range(num_train))
    split = int(val_split * num_train)
    # For reproducibility could shuffle here with seed
    train_idx, val_idx = indices[split:], indices[:split]

    # Create Subsets with appropriate transforms
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        
        def __len__(self):
            return len(self.subset)

    train_subset = torch.utils.data.Subset(full_train_data, train_idx)
    val_subset = torch.utils.data.Subset(full_train_data, val_idx)

    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

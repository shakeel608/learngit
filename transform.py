import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(batch_size=64):
    """
    Prepares the CIFAR-10 dataset with transforms and returns DataLoader objects for training and testing.
    
    Args:
        batch_size (int): Number of samples per batch.
    
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """
    # Define the transforms for training and testing datasets
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Augmentation: Randomly flip images
        transforms.RandomCrop(32, padding=4),  # Augmentation: Randomly crop images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader




print("Version 2 in progress....")

print("Version 3 in progress....")

print("Version 4 in progress....")

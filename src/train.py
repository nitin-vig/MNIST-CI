import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
import datetime

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data: a batch of images (64 images based on batch_size)
        # target: corresponding labels (correct digits) for these images
        
        data, target = data.to(device), target.to(device)
        # Move data to GPU if available, otherwise keep on CPU
        
        optimizer.zero_grad()
        # Reset gradients from previous batch
        
        output = model(data)
        # Forward pass: get model predictions
        
        loss = criterion(output, target)
        # Calculate loss between predictions and actual targets
        
        loss.backward()
        # Backward pass: calculate gradients
        
        optimizer.step()
        # Update model weights using calculated gradients
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'models/mnist_model_{timestamp}.pth')
    
if __name__ == "__main__":
    train()

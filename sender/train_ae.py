import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import logging
from utils.models import Autoencoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_autoencoder(epochs=10, batch_size=64, learning_rate=1e-3, save_path='./models/autoencoder_cifar10.pth'):
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Data (CIFAR-10)
    logging.info("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization needed if we want outputs in [0, 1] for sigmoid, 
        # but usually standardizing helps training. 
        # However, our Decoder ends with Sigmoid, so target should be [0, 1].
        # CIFAR-10 is naturally [0, 1] when loaded as Tensor.
    ])

    # We use the 'train' split for training
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 2. Initialize Model
    model = Autoencoder(encoded_space_dim=512).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training Loop
    logging.info(f"Starting training for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # 4. Save Model
    logging.info(f"Saving model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logging.info("Training complete!")

if __name__ == "__main__":
    # Ensure we are in the root directory or adjust paths
    # Assuming this is run from the project root where 'sender' and 'models' are accessible
    train_autoencoder()

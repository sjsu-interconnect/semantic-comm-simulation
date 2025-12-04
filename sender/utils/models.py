import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Convolutional Encoder for 32x32 images (CIFAR-10).
    Compresses image into a semantic vector.
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            # Input: 3 x 32 x 32
            nn.Conv2d(3, 16, 3, stride=2, padding=1), # -> 16 x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> 32 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # -> 64 x 4 x 4
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    """
    Convolutional Decoder for 32x32 images.
    Reconstructs image from semantic vector.
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 64 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(64, 4, 4))
        self.decoder_conv = nn.Sequential(
            # Input: 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # -> 32 x 8 x 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # -> 16 x 16 x 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)   # -> 3 x 32 x 32
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x) # Output range [0, 1]
        return x

class Autoencoder(nn.Module):
    """
    Combined Autoencoder for training or convenience.
    """
    def __init__(self, encoded_space_dim=512):
        super().__init__()
        self.encoder = Encoder(encoded_space_dim)
        self.decoder = Decoder(encoded_space_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

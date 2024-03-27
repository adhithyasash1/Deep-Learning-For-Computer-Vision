import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Define the generator model
        # A simple feedforward network for illustration purposes
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh(),
        )
    
    def forward(self, z, y):
        # Concatenate z and y and pass through the generator
        input = torch.cat([z, y], 1)
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Define the discriminator model
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, y):
        # Concatenate x and y and pass through the discriminator
        input = torch.cat([x, y], 1)
        return self.model(input)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Define the encoder model
        self.model_z = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 100),  # Output size: 100 for z
        )
        self.model_y = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),  # Output size: 10 for y
        )
    
    def forward(self, x):
        # Return latent representation z and conditional information y
        z = self.model_z(x)
        y = self.model_y(x)
        return z, y

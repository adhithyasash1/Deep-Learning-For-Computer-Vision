import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)  # Mean μ
        self.fc2 = nn.Linear(h_dim, z_dim)  # Log variance σ
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(h_dim, 64, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Output is an image
        )
        
    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, log_var)
        z = self.fc3(z).view(-1, 1024, 1, 1)
        return self.decoder(z), mu, log_var

# Model training
def train_vae(model, data_loader, epochs=5, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)
            optimizer.zero_grad()
            recon_x, mu, log_var = model(x)
            recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kld
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(data_loader.dataset)}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(dataset=mnist_data, batch_size=128, shuffle=True)

vae = VAE().to(device)
train_vae(vae, data_loader)

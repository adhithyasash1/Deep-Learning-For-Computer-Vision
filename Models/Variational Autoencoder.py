class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean of latent space
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent space
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function for VAE
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device (CUDA or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation pipeline for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to PyTorch tensors with values in [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalizes tensors to have values in [-1, 1]
])

# Download and load the training data for MNIST
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# DataLoader for batching and shuffling
train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)

# Optionally, load the test set
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=2)



vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

def train_vae(vae, train_loader, optimizer, epochs=10):
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

train_vae(vae, train_loader, optimizer)

class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x), 0.2)
        return self.fc2(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=256, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.leaky_relu(self.fc1(z), 0.2)
        return torch.sigmoid(self.fc2(h))

class Discriminator(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, z):
        h = F.leaky_relu(self.fc1(z), 0.2)
        return torch.sigmoid(self.fc2(h))


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


encoder = Encoder().to(device)
decoder = Decoder().to(device)
discriminator = Discriminator().to(device)

enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

def train_aae(encoder, decoder, discriminator, train_loader, enc_optimizer, dec_optimizer, disc_optimizer, epochs=10):
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            real_images = data.to(device)
            real_images_flat = real_images.view(real_images.size(0), -1)

            # === Train Discriminator ===
            z_real = torch.randn(batch_size, LATENT_SIZE).to(device)
            z_fake = encoder(real_images_flat)

            disc_real = discriminator(z_real)
            disc_fake = discriminator(z_fake.detach())

            disc_loss = discriminator_loss(disc_real, disc_fake)

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            # === Train Generator ===
            z_fake = encoder(real_images_flat)
            disc_fake = discriminator(z_fake)
            gen_loss = generator_loss(disc_fake)

            enc_optimizer.zero_grad()
            gen_loss.backward()
            enc_optimizer.step()

            # === Train Autoencoder ===
            recon_images = decoder(z_fake)
            recon_loss = F.binary_cross_entropy(recon_images, real_images_flat, reduction='sum') / real_images_flat.size(0)

            dec_optimizer.zero_grad()
            recon_loss.backward()
            dec_optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}')

train_aae(encoder, decoder, discriminator, train_loader, enc_optimizer, dec_optimizer, disc_optimizer)

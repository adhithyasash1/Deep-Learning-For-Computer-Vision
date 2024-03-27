class Generator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=784):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Ensures output is in the range [-1, 1]
        )
    
    def forward(self, x):
        return self.network(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability of being a real image
        )
    
    def forward(self, x):
        return self.network(x)


def adversarial_loss(disc_pred, is_real):
    target = torch.ones_like(disc_pred) if is_real else torch.zeros_like(disc_pred)
    return F.binary_cross_entropy(disc_pred, target)

def cycle_consistency_loss(real, reconstructed):
    return F.l1_loss(real, reconstructed)


def train_cycle_gan(gen_AB, gen_BA, disc_A, disc_B, loader_A, loader_B, num_epochs=10):
    # Optimizers
    opt_gen = torch.optim.Adam(itertools.chain(gen_AB.parameters(), gen_BA.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_disc_A = torch.optim.Adam(disc_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_disc_B = torch.optim.Adam(disc_B.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        for real_A, real_B in zip(loader_A, loader_B):
            real_A = real_A[0].to(device).view(real_A[0].size(0), -1)  # Assuming batch is the first element
            real_B = real_B[0].to(device).view(real_B[0].size(0), -1)
            
            # Generators AB and BA
            fake_B = gen_AB(real_A)
            fake_A = gen_BA(real_B)
            
            # Discriminator A
            opt_disc_A.zero_grad()
            loss_disc_A = adversarial_loss(disc_A(real_A), True) + adversarial_loss(disc_A(fake_A.detach()), False)
            loss_disc_A.backward()
            opt_disc_A.step()
            
            # Discriminator B
            opt_disc_B.zero_grad()
            loss_disc_B = adversarial_loss(disc_B(real_B), True) + adversarial_loss(disc_B(fake_B.detach()), False)
            loss_disc_B.backward()
            opt_disc_B.step()
            
            # Generator AB and BA
            opt_gen.zero_grad()
            loss_gen_AB = adversarial_loss(disc_B(fake_B), True)
            loss_gen_BA = adversarial_loss(disc_A(fake_A), True)
            
            # Cycle Consistency Loss
            recovered_A = gen_BA(fake_B)
            recovered_B = gen_AB(fake_A)
            loss_cycle_A = cycle_consistency_loss(real_A, recovered_A)
            loss_cycle_B = cycle_consistency_loss(real_B, recovered_B)
            
            loss_gen_total = loss_gen_AB + loss_gen_BA + 10 * (loss_cycle_A + loss_cycle_B)
            loss_gen_total.backward()
            opt_gen.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} completed.")


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformation pipeline for preprocessing and normalizing the datasets
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset as domain A
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader_A = DataLoader(mnist_train, batch_size=128, shuffle=True)

# Load Fashion-MNIST dataset as domain B
fashion_mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
loader_B = DataLoader(fashion_mnist_train, batch_size=128, shuffle=True)


# Instantiate the Cycle-GAN components
gen_AB = Generator(input_dim=784, hidden_dim=256, output_dim=784).to(device)
gen_BA = Generator(input_dim=784, hidden_dim=256, output_dim=784).to(device)
disc_A = Discriminator(input_dim=784, hidden_dim=256).to(device)
disc_B = Discriminator(input_dim=784, hidden_dim=256).to(device)

# Train the Cycle-GAN
train_cycle_gan(gen_AB, gen_BA, disc_A, disc_B, loader_A, loader_B, num_epochs=10)

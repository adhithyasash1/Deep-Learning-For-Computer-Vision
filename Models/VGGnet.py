import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class SimplifiedAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimplifiedAttention, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch, channels, height, width = x.size()
        attn = self.conv1x1(x).view(batch, 1, -1)
        attn = self.softmax(attn).view(batch, 1, height, width)
        return x * attn.expand_as(x)

class EfficientVGG(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientVGG, self).__init__()
        self.features = nn.Sequential(
            BasicConvBlock(3, 64),
            BasicConvBlock(64, 64, pool=True),
            BasicConvBlock(64, 128),
            BasicConvBlock(128, 128, pool=True),
            BasicConvBlock(128, 256),
            BasicConvBlock(256, 256, pool=True),
            BasicConvBlock(256, 512),
            BasicConvBlock(512, 512, pool=True),
            BasicConvBlock(512, 512),
            BasicConvBlock(512, 512, pool=True),
        )

        self.attention = nn.Sequential(
            SimplifiedAttention(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.classifier(x)
        return x

# Example of using the model
model = EfficientVGG(num_classes=10)  # Adjust the num_classes based on your dataset


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Creating data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the appropriate device
model = EfficientVGG(num_classes=10).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')

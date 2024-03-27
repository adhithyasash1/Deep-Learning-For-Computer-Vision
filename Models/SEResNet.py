class ConvolutionNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvolutionNet, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=7, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(5*5*8, num_classes)
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SEResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SEResNet, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU())
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(16))
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(16, 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4, 16, bias=False),
            nn.Sigmoid())
        
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(16*16*16, num_classes)
        
    def forward(self, x):
        x = self.conv_block1(x)
        identity = x
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        se = self.squeeze(x).view(-1, 16)
        se = self.excitation(se).view(-1, 16, 1, 1)
        x = x * se.expand_as(x)
        x += identity
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.prelu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return y + x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.layer1 = self.make_layer(ResidualBlock, 64, 2)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2)
        self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
        self.prelu2 = nn.PReLU(1024)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1000)
    def make_layer(self, block, channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)

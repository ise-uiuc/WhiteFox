
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, stride=1):
        super().__init__()
        self.blocks = []
        for idx in range(num_convs):
            if idx == 0:
                self.blocks.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride))
            else:
                self.blocks.append(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1))
    def forward(self, x1):
        y = 0
        for idx in range(len(self.blocks)):
            y = y + self.blocks[idx](x1)
        return y

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.conv2 = ConvBlock(8, 16, 2)
        self.conv3 = ConvBlock(16, 16, 1, 2)
        self.conv4 = ConvBlock(16, 32, 3)
        self.conv5 = ConvBlock(32, 8, 1)
        self.conv6 = ConvBlock(8, 16, 1, 2)
        self.conv7 = ConvBlock(16, 16, 3, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)

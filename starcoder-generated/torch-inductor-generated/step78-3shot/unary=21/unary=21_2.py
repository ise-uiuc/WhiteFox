
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution with kernel size 1
        self.conv1 = torch.nn.Conv2d(3, 64, 1, padding=0)
        # Depthwise convolution with kernel size 2
        self.conv2 = torch.nn.Conv2d(64, 64, 2, padding=0, groups=64)
        # Depthwise convolution with kernel size 2 and dilation of 3
        self.conv3 = torch.nn.Conv2d(64, 64, 2, padding=0, dilation=3, groups=64)
        # Pointwise convolution with kernel size 1
        self.conv4 = torch.nn.Conv2d(64, 128, 1, padding=0)
        # Pointwise convolution with kernel size 1
        self.conv5 = torch.nn.Conv2d(128, 256, 1, padding=0)
        # Pointwise convolution with kernel size 1
        self.conv6 = torch.nn.Conv2d(256, 512, 1, padding=0)
        # Pointwise convolution with kernel size 1
        self.conv7 = torch.nn.Conv2d(512, 1024, 1, padding=0)
        # Pointwise convolution with kernel size 1
        self.conv8 = torch.nn.Conv2d(1024, 10, 1, padding=0)
        # Batch norm
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.bn6 = torch.nn.BatchNorm2d(512)
        self.bn7 = torch.nn.BatchNorm2d(1024)
    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x))) - self.bn2(self.conv2(x))
        x = torch.asin(self.bn3(self.conv3(x))) + self.bn4(self.conv4(x))
        x = torch.relu(self.bn5(self.conv5(x))) - self.bn6(self.conv6(x))
        x = torch.sigmoid(self.bn7(self.conv7(x))) - self.bn8(self.conv8(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 28, 28)


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=1, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(16, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2, bias=False)
        self.gn2 = nn.GroupNorm(16, 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(4, 64)
        self.relu3 = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.gn3(x)
        x = self.relu3(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)

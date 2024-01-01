
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.LeakyReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(1, 128, (3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(128, 4, (3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True)
    def forward(self, x4):
        v1 = self.relu(self.conv1(x4))
        v2 = self.maxpool(v1)
        v3 = self.relu(self.conv2(v2))
        v4 = self.sigmoid(v3)
        return v4
# Inputs to the model
x4 = torch.randn(1, 1, 256, 256)

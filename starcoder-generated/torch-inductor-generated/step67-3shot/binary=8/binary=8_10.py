
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=[3, 3], stride=2)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0, bias=False)
        self.conv3 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0, bias=False)
    def forward(self, x):
        v1 = self.bn1(x)
        v2 = self.pool1(v1)
        v3 = self.relu1(v2)
        v4 = self.conv1(v3)
        v5 = self.bn2(v4)
        v6 = self.conv2(v5)
        v7 = self.conv3(v3)
        v8 = v4 + v6
        v9 = v8 * v7
        return v9
# Inputs to the model
x = torch.randn(1, 16, 224, 224)

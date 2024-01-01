
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3, dilation=2)
        self.conv2 = torch.nn.Conv2d(3, 48, 3, stride=2, padding=1, dilation=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(48)
        self.flatten = torch.nn.Flatten()
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.bn1(v1)
        v4 = self.bn2(v2)
        v5 = v3.add(v4)
        v6 = torch.tanh(v5)
        v7 = self.flatten(v6)
        v8 = v7 + v6
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)

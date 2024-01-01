
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 16, groups=4, kernel_size=3, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, groups=16, kernel_size=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, groups=64, kernel_size=1, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = self.bn1(v1)
        v3 = self.conv2(v2)
        v4 = self.bn2(v3)
        v5 = self.conv3(v4)
        v6 = self.bn3(v5)
        v7 = v6 * 0.044715
        v8 = v1 + v7
        v9 = v8 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v6 + 1
        v12 = v2 * v11
        return v12
# Inputs to the model
x2 = torch.randn(1, 32, 22, 77)



class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = v4 + v3
        v6 = v5 + v3
        return v6
# Inputs to the model
x = torch.randn(1, 3, 32, 32)

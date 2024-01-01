
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = self.bn2(v3)
        v6 = v2.add(v3)
        v7 = v2.add(v5)
        v8 = v1.add(v6)
        v9 = v1.add(v8)
        v10 = v9.add(v7)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 64, 64)

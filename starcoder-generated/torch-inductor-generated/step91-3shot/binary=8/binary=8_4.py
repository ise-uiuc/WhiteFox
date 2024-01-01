
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(torch.add(x1, x2))
        v2 = self.conv1(torch.add(x1, x2))
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = v4 + v3
        v6 = v5 + v2.add(v1)
        v7 = self.bn1(v5)
        v8 = v5 + v4
        v9 = v6 + v7
        return v8 + v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)

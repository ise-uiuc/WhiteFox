
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 * v2
        v4 = self.bn1(v1)
        v5 = v4 * v2
        v6 = self.bn2(v5)
        v7 = v3 + v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 8, 8)
x2 = torch.randn(1, 3, 8, 8)

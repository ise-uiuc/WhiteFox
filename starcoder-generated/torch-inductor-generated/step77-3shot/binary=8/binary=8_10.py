
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = v1.add(v2)
        v4 = self.conv2(v3)
        v5 = v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)

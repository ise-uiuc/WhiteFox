
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x1, x2):
        v1 = x1 + x2
        v2 = self.conv1(v1)
        v3 = self.conv2(v1)
        v4 = self.bn1(v2)
        v5 = self.bn2(v3)
        v6 = v4 - v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 2, 2)
x2 = torch.randn(1, 3, 2, 2)

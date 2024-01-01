
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU6()
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.bn1(self.relu(self.conv(x1)))
        v2 = v1 + 1
        v3 = torch.clamp(v2, 0, 6)
        v4 = self.bn2(torch.relu6(self.conv2(v3)))
        v5 = v4 + 1
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)

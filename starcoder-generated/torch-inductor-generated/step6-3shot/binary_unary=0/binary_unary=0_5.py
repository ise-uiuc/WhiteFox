
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2 + x2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = 1

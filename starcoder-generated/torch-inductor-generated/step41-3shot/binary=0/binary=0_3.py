
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 6, 7, stride=2, padding=5)
        self.bn = torch.nn.BatchNorm2d(6)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 56, 64, 64)

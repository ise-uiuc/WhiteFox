
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(16)
        self.conv = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(self.bn(x1))
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)

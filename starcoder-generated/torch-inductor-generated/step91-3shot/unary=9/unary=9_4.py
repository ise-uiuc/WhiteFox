
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = self.bn(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)

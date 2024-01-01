
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 128)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 300, 400)

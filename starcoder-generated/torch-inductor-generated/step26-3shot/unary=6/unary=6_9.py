
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.act = torch.nn.ReLU()
    def forward(self, x1):
        x1 = self.conv(x1)
        x2 = x1 + 3
        x3 = torch.clamp(x2, 0, 6)
        x4 = x1 * x3
        x5 = x4 / 6
        x6 = self.bn(x5)
        x7 = self.act(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)

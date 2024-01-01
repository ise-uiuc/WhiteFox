
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v = self.conv(x)
        return self.bn(v)
# Inputs to the model
x = torch.randn(1, 3, 32, 32)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 3, 1, 0)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x):
        return torch.nn.ReLU(self.bn(self.conv(x)) + self.bn(self.conv(x)))
# Inputs to the model
x = torch.randn(1, 2, 6, 6)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 2)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.bn(x)
        return x1, x2
# Inputs to the model
x = torch.randn(1, 2, 5, 5)

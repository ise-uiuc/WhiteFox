
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        return self.bn(self.conv(x))
# Inputs to the model
x = torch.randn(1, 4, 4, 4)

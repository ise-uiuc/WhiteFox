
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3)
        self.bn = torch.nn.BatchNorm2d(5)
    def forward(self, x3):
        x = self.conv(x3)
        y1 = x + x
        y = self.bn(y1)
        return y
# Inputs to the model
x3 = torch.randn(1, 3, 2, 2)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x1):
        y1 = self.bn(x1)
        if y1 is not None:
            y1 = self.conv(y1)
        return y1
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)

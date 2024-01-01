
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.bn1(x1)
        y2 = self.bn2(x1)
        return y2
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x2):
        s = self.conv(x2)
        t = self.bn1(s)
        y = self.bn2(t)
        return t
# Inputs to the model
x2 = torch.randn(1, 3, 6, 6)

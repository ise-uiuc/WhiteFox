
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.bn2 = torch.nn.BatchNorm2d(4)
    def forward(self, x):
        s = self.bn1(self.conv(x))
        return self.bn2(s)
# Inputs to the model
x = torch.randn(1, 4, 4, 4)

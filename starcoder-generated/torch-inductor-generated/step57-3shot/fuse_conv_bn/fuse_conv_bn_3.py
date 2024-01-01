
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 7, 3)
        self.bn = torch.nn.BatchNorm2d(7)
    def forward(self, x2):
        x2 = self.bn(self.conv(x2))
        return x2
# Inputs to the model
x2 = torch.randn(20, 8, 8, 8)

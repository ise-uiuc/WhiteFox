
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(8, 8, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(self.conv(y))
        return y
# Inputs to the model
x = torch.randn(1, 8, 16, 16)

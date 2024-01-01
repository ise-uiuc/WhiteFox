
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        y = self.conv(x)
        z = self.bn(y)
        return z
# Inputs to the model
x = torch.randn(1, 3, 4, 4)

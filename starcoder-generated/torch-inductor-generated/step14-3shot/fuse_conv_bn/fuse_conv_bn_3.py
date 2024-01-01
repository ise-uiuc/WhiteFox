
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.conv(x)
        y = self.bn(x)
        return torch.sum(y)
# Inputs to the model
x = torch.randn(1, 1, 8, 8)

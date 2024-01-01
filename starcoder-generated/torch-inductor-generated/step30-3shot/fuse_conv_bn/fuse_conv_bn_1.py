
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bias = torch.nn.Parameter(torch.randn(3))
        torch.manual_seed(0)
        self.bn = torch.nn.BatchNorm2d(3)
        self.bn.running_mean = torch.arange(3, dtype=torch.float)
        self.bn.running_var = torch.arange(3, dtype=torch.float) % 2 + 1
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 + self.bias
        x4 = self.bn(x3)
        return x2
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)

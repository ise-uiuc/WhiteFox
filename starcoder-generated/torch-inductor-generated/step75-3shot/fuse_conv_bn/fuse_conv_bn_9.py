
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm1d(3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.conv(v2)
        v4 = self.bn(v3)
        v5 = self.conv(v4)
        v6 = self.bn(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)

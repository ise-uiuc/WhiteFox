
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v2 = self.conv(v2)
        v2 = self.bn(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)

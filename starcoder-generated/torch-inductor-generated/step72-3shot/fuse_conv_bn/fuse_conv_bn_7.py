
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(1, 1, 2)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.conv(x1)
        v1 = self.bn(v2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)

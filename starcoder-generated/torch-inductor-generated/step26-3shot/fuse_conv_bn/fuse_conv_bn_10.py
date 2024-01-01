
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(2, 2, 2)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        v1 = self.bn(x1)
        v1 = self.conv(v1)
        v1 = self.bn(v1)
        v1 = self.conv(v1)
        v1 = self.bn(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)

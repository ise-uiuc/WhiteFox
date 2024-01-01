
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = self.bn(x2)
        x4 = x3 + 3.14
        x5 = F.relu(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 64)
